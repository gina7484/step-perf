# ======================= HBM Store =======================
event_name="SimpleEvent"
db_name="test_prefill_expert_mnk_mnk_0613"
# event_name="WeightQLoad"
# event_name="GenQ"
# event_name="StoreOutput"
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
