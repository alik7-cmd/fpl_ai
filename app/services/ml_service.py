import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

class MLService:
    POSITION_FEATURES = {
        "GK": ["minutes", "saves", "clean_sheets", "goals_conceded", "penalties_saved",
               "yellow_cards", "red_cards", "form", "points_per_game", "fdr", "availability"],
        "DEF": ["minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
                "yellow_cards", "red_cards", "form", "points_per_game", "fdr", "availability"],
        "MID": ["minutes", "goals_scored", "assists", "creativity", "influence", "threat", "ict_index",
                "penalty_taker", "set_piece_taker", "yellow_cards", "red_cards",
                "form", "points_per_game", "fdr", "availability"],
        "FWD": ["minutes", "goals_scored", "assists", "threat", "influence", "ict_index",
                "penalty_taker", "set_piece_taker", "yellow_cards", "red_cards",
                "form", "points_per_game", "fdr", "availability"]
    }

    # ------------------ Evaluation & Scoring ------------------ #
    @staticmethod
    def evaluate_model(model, X_val, y_val):
        y_pred = model.predict(X_val)
        return {
            "R2": r2_score(y_val, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_val, y_pred)),
            "MAE": mean_absolute_error(y_val, y_pred)
        }

    @staticmethod
    def compute_model_score(metrics):
        # Higher is better
        r2 = metrics["R2"]
        rmse_score = 1 / (metrics["RMSE"] + 1e-6)  # avoid division by zero
        mae_score = 1 / (metrics["MAE"] + 1e-6)
        return (r2 + rmse_score + mae_score) / 3

    # ------------------ Model Training ------------------ #
    @staticmethod
    def train_models(players, fdr_map):
        df = pd.DataFrame(players)
        df["fdr"] = df["team"].map(fdr_map).fillna(3)
        df["element_type"] = df["element_type"].map({1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}).fillna("UNK")

        # Encode special roles
        df["penalty_taker"] = df["penalties_order"].apply(lambda x: 1 if x else 0)
        df["set_piece_taker"] = df[["corners_and_indirect_freekicks_order", "direct_freekicks_order"]].apply(
            lambda row: 1 if row.notnull().any() else 0, axis=1)

        # Fill numeric columns
        numeric_cols = ["minutes", "points_per_game", "form", "goals_scored", "assists",
                        "clean_sheets", "goals_conceded", "penalties_saved", "yellow_cards",
                        "red_cards", "saves", "influence", "creativity", "threat",
                        "ict_index", "penalty_taker", "set_piece_taker"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["availability"] = df.get("chance_of_playing_next_round", 100).fillna(100) / 100
        df["y"] = df["total_points"] / (df["minutes"] / 90 + 0.01)

        best_models = {}

        for pos, feats in MLService.POSITION_FEATURES.items():
            dpos = df[df["element_type"] == pos]
            if len(dpos) < 5:
                continue

            X = dpos[feats]
            y = dpos["y"]
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            model_dict = {}
            metric_dict = {}

            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            model_dict["Linear"] = lr
            metric_dict["Linear"] = MLService.evaluate_model(lr, X_val, y_val)

            # XGBoost
            xgb = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.1, random_state=42)
            xgb.fit(X_train, y_train)
            model_dict["XGB"] = xgb
            metric_dict["XGB"] = MLService.evaluate_model(xgb, X_val, y_val)

            # RandomForest
            rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
            rf.fit(X_train, y_train)
            model_dict["RandomForest"] = rf
            metric_dict["RandomForest"] = MLService.evaluate_model(rf, X_val, y_val)

            # CatBoost
            cb = CatBoostRegressor(n_estimators=100, depth=5, learning_rate=0.1, verbose=0, random_state=42)
            cb.fit(X_train, y_train)
            model_dict["CatBoost"] = cb
            metric_dict["CatBoost"] = MLService.evaluate_model(cb, X_val, y_val)

            # Robust best model selection
            best_model_name = max(metric_dict, key=lambda k: MLService.compute_model_score(metric_dict[k]))
            best_models[pos] = (model_dict[best_model_name], feats, best_model_name)

        return best_models

    # ------------------ Prediction ------------------ #
    @staticmethod
    def predict_points(player, fdr_map, models):
        chance = player.get("chance_of_playing_next_round", 100) or 100
        if chance < 50:
            return 0.0

        pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        pos = pos_map.get(player.get("element_type"), "UNK")
        if pos not in models:
            return 0.0

        model, feats, model_name = models[pos]

        # Encode special roles
        player["penalty_taker"] = 1 if player.get("penalties_order") else 0
        player["set_piece_taker"] = 1 if (player.get("corners_and_indirect_freekicks_order") or
                                          player.get("direct_freekicks_order")) else 0

        # Build feature row
        row = {}
        for f in feats:
            if f == "fdr":
                row[f] = fdr_map.get(player["team"], 3)
            elif f == "availability":
                row[f] = chance / 100
            else:
                row[f] = float(player.get(f, 0))

        dfp = pd.DataFrame([row])
        pred = model.predict(dfp[feats])[0]

        # Adjust for availability
        pred *= row["availability"]

        # Discipline penalty
        discipline_penalty = 1 - 0.1 * row.get("yellow_cards", 0) - 0.3 * row.get("red_cards", 0)
        pred *= max(discipline_penalty, 0)

        # Extra points for MID/FWD
        if pos in ["MID", "FWD"]:
            pred += 0.2 * row.get("penalty_taker", 0) + 0.1 * row.get("set_piece_taker", 0)

        # Scale by minutes played
        pred *= min(float(player.get("minutes", 0)) / 90, 1)

        return round(pred, 2)
