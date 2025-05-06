import os
from config import config
from evaluation.evaluation_pipeline import evaluate_configurations, get_scenarios
from database.dataManager import load_and_prepare_prices
from datetime import timedelta, datetime

def test_evaluation():


    # 2. Evaluate all scenarios
    print("Starting evaluation...")
    evaluate_configurations()

    print("Evaluation finished!")

