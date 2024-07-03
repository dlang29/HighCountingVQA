from utils import get_csv, calculate_metrics, write_to_csv
from plots import create_data_distribution

create_data_distribution("./data/HighCountVQA_combined.json")
create_data_distribution("./data/HighCountVQA_test.json")
create_data_distribution("./data/HighCountVQA_val.json")
"""
results_df = get_csv("results", "google/paligemma-3b-mix-224")
bin_df = calculate_metrics(results_df)
print(bin_df)
write_to_csv(bin_df, "bin")
"""