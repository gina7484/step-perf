python scripts/gantt_chart_generator.py --csv_files \
    data/hbm_load_gen_qkv.csv \
    data/load_gen_qkv.csv \
    data/comp_gen_qkv.csv \
    \
    data/hbm_load_q_kt.csv \
    data/load_q_kt.csv \
    data/comp_q_kt.csv \
    \
    data/hbm_load_attn_v.csv \
    data/load_attn_v.csv \
    data/comp_attn_v.csv \
    \
    data/hbm_load_proj.csv \
    data/load_proj.csv \
    data/comp_proj.csv \
    data/store_proj.csv \
    \
    data/hbm_store.csv \
    --output_file gantt_chart.html