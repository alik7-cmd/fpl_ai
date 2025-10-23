from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from typing import Optional, List

from app.core.config import POSITIONS, BASE_IMAGE_URL
from app.services.data_service import FPLDataService
from app.services.ml_service import MLService
from app.services.optimizer_service import OptimizerService

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/top")
def top_players(
    n: int = Query(5, gt=0, description="Number of top players per position"),
    position: Optional[str] = Query(None, description="Position filter: GK, DEF, MID, FWD")
):
    position = position.upper() if position else None
    valid_positions = ["GK", "DEF", "MID", "FWD"]
    if position and position not in valid_positions:
        return {"error": f"Invalid position '{position}'. Must be one of {valid_positions}"}

    players, teams, fixtures, gw = FPLDataService.fetch_fpl_data()
    team_map = {t["id"]: t["name"] for t in teams}
    fdr_map = FPLDataService.get_team_fdr(fixtures)

    models = MLService.train_models(players, fdr_map)

    enriched = []
    for p in players:
        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        pos = pos_map.get(p.get("element_type"), "UNK")

        # Only MID/FWD get penalty/set-piece
        if pos in ["MID", "FWD"]:
            p["penalty_taker"] = 1 if p.get("penalties_order") else 0
            p["set_piece_taker"] = 1 if (p.get("corners_and_indirect_freekicks_order") or
                                         p.get("direct_freekicks_order")) else 0
        else:
            p["penalty_taker"] = 0
            p["set_piece_taker"] = 0

        # Common features
        p["yellow_cards"] = p.get("yellow_cards", 0)
        p["red_cards"] = p.get("red_cards", 0)
        p["availability"] = (p.get("chance_of_playing_next_round", 100) or 100) / 100

        ep = MLService.predict_points(p, fdr_map, models)
        if ep <= 0:
            continue

        enriched.append({
            "name": f"{p['first_name']} {p['second_name']}",
            "team": team_map[p["team"]],
            "position": POSITIONS[p["element_type"]],
            "price": p["now_cost"] / 10,
            "expected_points": ep,
            "minutes": p.get("minutes", 0),
            "chance": p.get("chance_of_playing_next_round", 100),
            "penalty_taker": p["penalty_taker"],
            "set_piece_taker": p["set_piece_taker"],
            "yellow_cards": p["yellow_cards"],
            "red_cards": p["red_cards"],
            "availability": p["availability"],
            "image": BASE_IMAGE_URL + p.get("photo", "")
        })

    positions_to_return = [position] if position else valid_positions
    top_n = {}
    for pos in positions_to_return:
        pos_players = [p for p in enriched if p["position"] == pos]
        sorted_players = sorted(pos_players, key=lambda x: x["expected_points"], reverse=True)[:n]
        top_n[pos] = sorted_players

    return {
        "gameweek": gw,
        "top_players": top_n
    }


@app.get("/team")
def build_team():
    players, teams, fixtures, gw = FPLDataService.fetch_fpl_data()
    team_map = {t["id"]: t["name"] for t in teams}
    fdr_map = FPLDataService.get_team_fdr(fixtures)

    models = MLService.train_models(players, fdr_map)

    enriched = []
    for p in players:
        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        pos = pos_map.get(p.get("element_type"), "UNK")

        # Only MID/FWD get penalty/set-piece
        if pos in ["MID", "FWD"]:
            p["penalty_taker"] = 1 if p.get("penalties_order") else 0
            p["set_piece_taker"] = 1 if (p.get("corners_and_indirect_freekicks_order") or
                                         p.get("direct_freekicks_order")) else 0
        else:
            p["penalty_taker"] = 0
            p["set_piece_taker"] = 0

        # Common features
        p["yellow_cards"] = p.get("yellow_cards", 0)
        p["red_cards"] = p.get("red_cards", 0)
        p["availability"] = (p.get("chance_of_playing_next_round", 100) or 100) / 100

        ep = MLService.predict_points(p, fdr_map, models)
        if ep <= 0:
            continue

        pos_name = pos_map.get(p.get("element_type"), "UNK")
        model_name = models[pos_name][2] if pos_name in models else "Unknown"


        enriched.append({
            "name": f"{p['first_name']} {p['second_name']}",
            "team": team_map[p["team"]],
            "position": POSITIONS[p["element_type"]],
            "price": p["now_cost"] / 10,
            "expected_points": ep,
            "minutes": p.get("minutes", 0),
            "chance": p.get("chance_of_playing_next_round", 100),
            "penalty_taker": p["penalty_taker"],
            "set_piece_taker": p["set_piece_taker"],
            "yellow_cards": p["yellow_cards"],
            "red_cards": p["red_cards"],
            "availability": p["availability"],
            "model_used": model_name
           
        })

    optimizer = OptimizerService()
    squad = optimizer.optimize_team(enriched)
    xi, bench, formation, xi_points = optimizer.pick_xi(squad)
    captain, vice = optimizer.pick_captain(xi)
    bench_points = sum(p["expected_points"] for p in bench)

    if captain:
        xi_points += captain["expected_points"]

    return {
        "gameweek": gw,
        "starting_formation": formation,
        "starting_xi": optimizer.group_pos(xi),
        "bench": optimizer.group_pos(bench),
        "captain": captain,
        "vice_captain": vice,
        "starting_xi_points": xi_points,
        "bench_points": bench_points,
        "total_team_cost": sum(p["price"] for p in squad)
    }

@app.get("/player/impact")
def player_impact(player_id: int):
    """Return feature contributions for a single player."""
    players, teams, fixtures, gw = FPLDataService.fetch_fpl_data()
    fdr_map = FPLDataService.get_team_fdr(fixtures)
    models = MLService.train_models(players, fdr_map)

    player = next((p for p in players if p["id"] == player_id), None)
    if not player:
        return {"error": "Player not found"}

    # Enrich player
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    pos = pos_map.get(player.get("element_type"), "UNK")

    player["penalty_taker"] = 1 if pos in ["MID", "FWD"] and player.get("penalties_order") else 0
    player["set_piece_taker"] = 1 if pos in ["MID", "FWD"] and (player.get("corners_and_indirect_freekicks_order") or player.get("direct_freekicks_order")) else 0
    player["yellow_cards"] = player.get("yellow_cards", 0)
    player["red_cards"] = player.get("red_cards", 0)
    player["availability"] = (player.get("chance_of_playing_next_round", 100) or 100) / 100

    predicted_points = MLService.predict_points(player, fdr_map, models)
    pos_name = pos_map.get(player.get("element_type"), "UNK")
    model_name = models[pos_name][2] if pos_name in models else "Unknown"

    # Feature breakdown (simple proportional contribution based on features)
    feature_impact = {
        "goals_scored": player.get("goals_scored", 0),
        "assists": player.get("assists", 0),
        "clean_sheets": player.get("clean_sheets", 0) if pos in ["GK","DEF"] else 0,
        "saves": player.get("saves", 0) if pos == "GK" else 0,
        "penalty_set_piece": player["penalty_taker"] + player["set_piece_taker"],
        "cards": -(player["yellow_cards"]*0.5 + player["red_cards"]*1.0),
        "availability": player["availability"]
    }

    return {
        "player": f"{player['first_name']} {player['second_name']}",
        "position": POSITIONS.get(player["element_type"], "UNK"),
        "expected_points": predicted_points,
        "feature_impact": feature_impact,
        "model_used": model_name
    }

@app.get("/player/performance-trends")
def player_performance_trends(player_id: int, n_gameweeks: int = 5):
    """Return historical trends for a player."""
    players, teams, fixtures, gw = FPLDataService.fetch_fpl_data()
    fdr_map = FPLDataService.get_team_fdr(fixtures)
    models = MLService.train_models(players, fdr_map)

    player = next((p for p in players if p["id"] == player_id), None)
    if not player:
        return {"error": "Player not found"}

    trends = []
    # Use last n_gameweeks or mocked data from total_points / minutes
    # Here we simulate; in a real app fetch from historical API or database
    for i in range(n_gameweeks):
        trends.append({
            "gw": gw-i,
            "actual_points": max(0, player.get("total_points",0)//(i+1)),
            "predicted_points": round(MLService.predict_points(player, fdr_map, models),2)
        })
    return {
        "player": f"{player['first_name']} {player['second_name']}",
        "position": POSITIONS.get(player["element_type"], "UNK"),
        "trends": trends
    }

@app.get("/team/risk")
def team_risk(team_ids: str = Query(..., description="Comma-separated list of player IDs")):
    """Return risk scores for players based on availability, cards, and rotation."""
    players, teams, fixtures, gw = FPLDataService.fetch_fpl_data()
    fdr_map = FPLDataService.get_team_fdr(fixtures)
    models = MLService.train_models(players, fdr_map)

    team_ids = [int(x) for x in team_ids.split(",")]

    result = []
    for pid in team_ids:
        p = next((pl for pl in players if pl["id"] == pid), None)
        if not p:
            continue
        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        pos = pos_map.get(p.get("element_type"), "UNK")

        availability = (p.get("chance_of_playing_next_round", 100) or 100) / 100
        cards_score = (p.get("yellow_cards", 0)*0.1 + p.get("red_cards", 0)*0.3)
        rotation_risk = 0.2 if p.get("minutes", 0) < 60 else 0

        risk_score = 0.5*(1-availability) + 0.3*cards_score + rotation_risk

        result.append({
            "name": f"{p['first_name']} {p['second_name']}",
            "position": POSITIONS.get(p["element_type"], "UNK"),
            "risk_score": round(risk_score,2),
            "factors": {
                "availability": round(0.5*(1-availability),2),
                "cards": round(0.3*cards_score,2),
                "rotation": round(rotation_risk,2)
            }
        })
    return {"team_risk": result}

@app.get("/team/impact-summary")
def team_impact_summary(team_ids: str = Query(..., description="Comma-separated list of player IDs")):
    """Aggregate feature contributions across a squad."""
    players, teams, fixtures, gw = FPLDataService.fetch_fpl_data()
    fdr_map = FPLDataService.get_team_fdr(fixtures)
    models = MLService.train_models(players, fdr_map)

    total_points = 0
    summary = {
        "goals":0, "assists":0, "clean_sheets":0, "saves":0, "penalty_set_piece":0, "cards":0, "availability":0
    }
    team_ids = [int(x) for x in team_ids.split(",")]

    for pid in team_ids:
        p = next((pl for pl in players if pl["id"] == pid), None)
        if not p: continue
        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        pos = pos_map.get(p.get("element_type"), "UNK")

        p["penalty_taker"] = 1 if pos in ["MID", "FWD"] and p.get("penalties_order") else 0
        p["set_piece_taker"] = 1 if pos in ["MID", "FWD"] and (p.get("corners_and_indirect_freekicks_order") or p.get("direct_freekicks_order")) else 0
        p["yellow_cards"] = p.get("yellow_cards",0)
        p["red_cards"] = p.get("red_cards",0)
        p["availability"] = (p.get("chance_of_playing_next_round",100) or 100)/100

        ep = MLService.predict_points(p, fdr_map, models)
        total_points += ep

        summary["goals"] += p.get("goals_scored",0)
        summary["assists"] += p.get("assists",0)
        summary["clean_sheets"] += p.get("clean_sheets",0) if pos in ["GK","DEF"] else 0
        summary["saves"] += p.get("saves",0) if pos=="GK" else 0
        summary["penalty_set_piece"] += p["penalty_taker"] + p["set_piece_taker"]
        summary["cards"] += p["yellow_cards"] + 2*p["red_cards"]
        summary["availability"] += p["availability"]

    return {
        "total_predicted_points": round(total_points,2),
        "feature_impact_summary": summary
    }
