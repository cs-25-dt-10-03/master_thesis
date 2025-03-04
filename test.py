import flexoffer_logic

# Create a TimeSlice
ts1 = flexoffer_logic.TimeSlice(1.0, 2.0)

ts2 = flexoffer_logic.TimeSlice(2.0, 3.0)

# Create a Flexoffer
fo = flexoffer_logic.Flexoffer(1, 1, 2, 4, [ts1, ts2], 2)

print(fo.get_offer_id())
print(fo.get_est_hour())
print(fo.get_total_energy()) 

