# ======================= HBM Store =======================
event_name="InputLoad"
# event_name="WeightQLoad"
# event_name="GenQ"
# event_name="StoreOutput"
mongosh --quiet test_mm --eval 'print("timestamp,start_ns,end_ns,is_stop"); db.log.aggregate([
  { $match: { event_type: "'${event_name}'" } },
  { $project: { 
      _id: 0, 
      timestamp: 1, 
      start_ns: "$event_data.start_ns", 
      end_ns: "$event_data.end_ns",
      is_stop: "$event_data.is_stop",
    }
  },
  { $sort: { timestamp: 1 } }
]).forEach(function(doc) { 
  print(doc.timestamp + "," + doc.start_ns + "," + doc.end_ns + "," + doc.is_stop); 
})' > data/${event_name}.csv
