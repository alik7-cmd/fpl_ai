from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary
from app.core.config import TOTAL_BUDGET, VALID_FORMATIONS

class OptimizerService:
    @staticmethod
    def optimize_team(players):
        prob = LpProblem("FPL_Optimization", LpMaximize)
        vars_ = {p["name"]: LpVariable(p["name"], cat=LpBinary) for p in players}

        prob += lpSum(vars_[p["name"]] * p["expected_points"] for p in players)
        prob += lpSum(vars_[p["name"]] * p["price"] for p in players) <= TOTAL_BUDGET
        prob += lpSum(vars_[p["name"]] for p in players) == 15

        pos_limits = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
        for pos, n in pos_limits.items():
            prob += lpSum(vars_[p["name"]] for p in players if p["position"] == pos) == n

        for team in set(p["team"] for p in players):
            prob += lpSum(vars_[p["name"]] for p in players if p["team"] == team) <= 3

        prob.solve()
        return [p for p in players if vars_[p["name"]].varValue == 1]

    @staticmethod
    def pick_xi(squad):
        best_points = 0
        best_xi = None
        best_form = None

        pos_groups = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for p in squad:
            pos_groups[p["position"]].append(p)

        for form in VALID_FORMATIONS:
            xi = (
                sorted(pos_groups["GK"], key=lambda x: x["expected_points"], reverse=True)[:1]
                + sorted(pos_groups["DEF"], key=lambda x: x["expected_points"], reverse=True)[:form["DEF"]]
                + sorted(pos_groups["MID"], key=lambda x: x["expected_points"], reverse=True)[:form["MID"]]
                + sorted(pos_groups["FWD"], key=lambda x: x["expected_points"], reverse=True)[:form["FWD"]]
            )
            total = sum(p["expected_points"] for p in xi)
            if total > best_points:
                best_points = total
                best_xi = xi
                best_form = form

        bench = [p for p in squad if p not in best_xi]
        return best_xi, bench, best_form, best_points

    @staticmethod
    def pick_captain(xi):
        safe = sorted([p for p in xi if p["expected_points"] > 0], 
                     key=lambda x: x["expected_points"], reverse=True)
        cap = safe[0] if safe else None
        vice = safe[1] if len(safe) > 1 else None
        return cap, vice

    @staticmethod
    def group_pos(players):
        grouped = {"GK": [], "DEF": [], "MID": [], "FWD": []}
        for p in players:
            grouped[p["position"]].append(p)
        return grouped
