mongosh --quiet init_hbm_log_generic --eval 'print("timestamp,start(ns),end(ns)"); db.log.aggregate([
  { $match: { event_type: "GenQKV" } },
  { $project: { 
      _id: 0, 
      timestamp: 1, 
      start: "$event_data.start", 
      end: "$event_data.end" 
    }
  },
  { $sort: { timestamp: 1 } }
]).forEach(function(doc) { 
  print(doc.timestamp + "," + doc.start + "," + doc.end); 
})' > init_hbm_log_generic.csv