# Config for Llama 3.1 70B
H = 8192  # hidden size
N_LAYERS = 80
N_HEADS = 64
HEAD_DIM = H // N_HEADS
MLP_HID = 22016

# Datatype
N_BYTE = 2  # fp16

# Batch Info
BATCH_LENGTHS = [
    374,
    396,
    879,
    91,
    91,
    381,
    1313,
    388,
    242,
    209,
    394,
    394,
    1315,
    2221,
    389,
    415,
    120,
    369,
    206,
    1353,
    197,
    181,
    388,
    4085,
]
BATCH_SIZE = len(BATCH_LENGTHS)
SEQ_LEN_SUM = sum(BATCH_LENGTHS)

EXPANDED_BATCH_LENGTHS = [length for length in BATCH_LENGTHS for _ in range(N_HEADS)]
SN40L_COMP = 638  # TFLOP/s
SN40L_MEM_BW = 1.8  # TB/s


gen_qkv_flop = 1 * H * 3 * H
q_kt_flop = [
    N_HEADS * HEAD_DIM * seq_len + N_HEADS * seq_len
    for seq_len in EXPANDED_BATCH_LENGTHS
]
attn_v_flop = [N_HEADS * HEAD_DIM * seq_len for seq_len in EXPANDED_BATCH_LENGTHS]
proj_flop = 1 * H * H

print(gen_qkv_flop)
print(q_kt_flop)
print(attn_v_flop)
print(proj_flop)
