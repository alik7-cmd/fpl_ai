import requests
from app.core.config import BASE_API_URL

class FPLDataService:
    @staticmethod
    def fetch_fpl_data():
        res = requests.get(BASE_API_URL + "bootstrap-static/")
        res.raise_for_status()
        data = res.json()

        players = data["elements"]
        teams = data["teams"]
        fixtures = requests.get(BASE_API_URL + "fixtures/").json()
        events = data["events"]
        next_gw = next((e for e in events if not e["finished"]), None)
        next_gw_num = next_gw["id"] if next_gw else None

        return players, teams, fixtures, next_gw_num

    @staticmethod
    def get_team_fdr(fixtures):
        fdr = {}
        for f in fixtures:
            if not f["finished"]:
                fdr.setdefault(f["team_h"], []).append(f["team_h_difficulty"])
                fdr.setdefault(f["team_a"], []).append(f["team_a_difficulty"])
        return {tid: sum(vals)/len(vals) for tid, vals in fdr.items() if vals}
