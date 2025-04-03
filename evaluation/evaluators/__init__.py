from .optimizer import run_optimizer
from .baselines import run_greedy, run_naive
from evaluation.metrics.core import evaluate_schedule

from flexoffer_logic import Flexoffer, DFO

from evaluation.utils.price_loader import get_hourly_spot_prices, get_hourly_mfrr_prices
from evaluation.metrics.market import estimate_mfrr_up_revenue

def run_all_evaluators(obj):
    start_time = 1711929600  # placeholder 
    duration = 24  # hours

    spot_prices = get_hourly_spot_prices(start_time, duration)
    mfrr = get_hourly_mfrr_prices(start_time, duration)

    results = {}

    # Optimizer
    schedule_opt = run_optimizer(obj, spot_prices)
    eval_opt = evaluate_schedule(schedule_opt, spot_prices)
    eval_opt["mFRR_up_revenue"] = estimate_mfrr_up_revenue(schedule_opt, mfrr["up"])
    results["optimizer"] = eval_opt

    # Greedy
    schedule_greedy = run_greedy(obj, spot_prices)
    eval_greedy = evaluate_schedule(schedule_greedy, spot_prices)
    eval_greedy["mFRR_up_revenue"] = estimate_mfrr_up_revenue(schedule_greedy, mfrr["up"])
    results["greedy"] = eval_greedy

    # Naive
    schedule_naive = run_naive(obj, spot_prices)
    eval_naive = evaluate_schedule(schedule_naive, spot_prices)
    eval_naive["mFRR_up_revenue"] = estimate_mfrr_up_revenue(schedule_naive, mfrr["up"])
    results["naive"] = eval_naive

    return results
