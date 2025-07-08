# ======================= HBM Store =======================
event_name="SimpleEvent"

db_names=(
    # "expert_mn_mk_32_256_32_256"
    # "expert_mn_mk_32_256_64_256"
    # "expert_mn_mk_64_256_128_256"
    "expert_mn_mk_64_256_16_256"
    "expert_mn_mk_64_256_256_256"
    "expert_mn_mk_64_256_32_256"
    "expert_mn_mk_64_256_64_256"
)


# Loop through each database name
for db_name in "${db_names[@]}"; do
  echo "Processing database: ${db_name}"
  mongosh --quiet ${db_name} --eval 'print("timestamp,name,id,start_ns,end_ns,is_stop"); db.log.aggregate([
    { $match: { event_type: "'${event_name}'" } },
    { $project: { 
        _id: 0, 
        timestamp: 1, 
        name: "$event_data.name",
        id: "$event_data.id", 
        start_ns: "$event_data.start_ns", 
        end_ns: "$event_data.end_ns",
        is_stop: "$event_data.is_stop",
      }
    },
    { $sort: { timestamp: 1 } }
  ]).forEach(function(doc) { 
    print(doc.timestamp + "," + doc.name + "," + doc.id + "," + doc.start_ns + "," + doc.end_ns + "," + doc.is_stop); 
  })' > data/${db_name}.csv
  echo "Exported data to data/${db_name}.csv"
done