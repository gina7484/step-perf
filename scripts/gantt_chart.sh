db_names=(
    # "expert_mn_mk_16_256_128_256"
    # "expert_mn_mk_16_256_256_256"
    # "expert_mn_mk_32_256_16_256"
    # "expert_mn_mk_32_256_32_256"
    # "expert_mn_mk_32_256_64_256"
    # "expert_mn_mk_64_256_128_256"
    # "expert_mn_mk_64_256_16_256"
    # "expert_mn_mk_64_256_256_256"
    # "expert_mn_mk_64_256_32_256"
    # "expert_mn_mk_64_256_64_256"
    # "mlp_2d"
    # "mlp_2d_delete_red_buff"
    "mlp_2d_merge_offchip"
)

# Loop through each database name
for db_name in "${db_names[@]}"; do

    python scripts/gantt_chart_generator.py \
    --csv_file data/${db_name}.csv \
    --output_file data/gantt_chart_${db_name}.html

    echo "Generated data/gantt_chart_${db_name}.html"
done