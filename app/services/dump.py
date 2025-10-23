# from fastapi import FastAPI, Query
# from typing import Optional, List

# from app.core.config import POSITIONS, BASE_IMAGE_URL
# from app.services.data_service import FPLDataService
# from app.services.ml_service import MLService

# app = FastAPI()







# @app.get("/ml/models")
# def ml_models():
#     """Return features used and metadata for position-specific models."""
#     positions = ["GK", "DEF", "MID", "FWD"]
#     feature_metadata = {}
#     for pos in positions:
#         if pos in ["MID","FWD"]:
#             features = ["minutes","goals_scored","assists","creativity","influence","threat","ict_index",
#                         "penalty_taker","set_piece_taker","yellow_cards","red_cards","form","points_per_game",
#                         "fdr","availability"]
#         elif pos=="DEF":
#             features = ["minutes","goals_scored","assists","clean_sheets","goals_conceded","yellow_cards",
#                         "red_cards","form","points_per_game","fdr","availability"]
#         else: # GK
#             features = ["minutes","saves","clean_sheets","goals_conceded","penalties_saved","yellow_cards",
#                         "red_cards","form","points_per_game","fdr","availability"]
#         feature_metadata[pos] = {
#             "features": features,
#             "model_type": ["LinearRegression","XGBRegressor"]
#         }
#     return {"models": feature_metadata}






