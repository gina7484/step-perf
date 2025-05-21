# ======================= HBM Store =======================
mongosh --quiet attn_log --eval 'print("timestamp,m,n,k,start_ns,end_ns"); db.log.aggregate([
  { $match: { event_type: "HBMOutput" } },
  { $project: { 
      _id: 0, 
      timestamp: 1, 
      m: "$event_data.m", 
      n: "$event_data.n",
      k: "$event_data.k",
      start_ns: "$event_data.start_ns",
      end_ns: "$event_data.end_ns"
    }
  },
  { $sort: { timestamp: 1 } }
]).forEach(function(doc) { 
  print(doc.timestamp + "," + doc.m + "," + doc.n + "," + doc.k + "," + doc.start_ns + "," + doc.end_ns); 
})' > data/hbm_store.csv
