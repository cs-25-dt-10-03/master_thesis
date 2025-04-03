
def estimate_mfrr_up_revenue(schedule, up_prices, threshold=0.5):
    revenue = 0.0
    for e, p in zip(schedule, up_prices):
        if e > threshold:
            revenue += (e - threshold) * p
    return revenue

def estimate_mfrr_down_cost(schedule, down_prices, threshold=0.5):
    cost = 0.0
    for e, p in zip(schedule, down_prices):
        if e < threshold:
            cost += (threshold - e) * p
    return cost
