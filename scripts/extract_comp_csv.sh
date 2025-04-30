
# ======================= Compute =======================
## GenQKV

mongosh --quiet attn_log --eval 'print("timestamp,counter,start,end"); db.log.aggregate([
  { $match: { event_type: "CompGenQKV" } },
  { $project: { 
      _id: 0, 
      timestamp: 1, 
      counter: "$event_data.counter", 
      start: "$event_data.start", 
      end: "$event_data.end"
    }
  },
  { $sort: { timestamp: 1 } }
]).forEach(function(doc) { 
  print(doc.timestamp + "," + doc.counter + "," + doc.start + "," + doc.end); 
})' > comp_gen_qkv.csv 

mongosh --quiet attn_log --eval 'print("timestamp,start,end"); db.log.aggregate([
  { $match: { event_type: "LoadQKV" } },
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
})' > load_gen_qkv.csv 

## Q_KT
mongosh --quiet attn_log --eval 'print("timestamp,start,end"); db.log.aggregate([
  { $match: { event_type: "CompQKt" } },
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
})' > comp_q_kt.csv 

mongosh --quiet attn_log --eval 'print("timestamp,start,end"); db.log.aggregate([
  { $match: { event_type: "LoadQKt" } },
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
})' > load_q_kt.csv 

## Attn_v

mongosh --quiet attn_log --eval 'print("timestamp,start,end"); db.log.aggregate([
  { $match: { event_type: "CompAttnV" } },
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
})' > comp_attn_v.csv 

mongosh --quiet attn_log --eval 'print("timestamp,start,end"); db.log.aggregate([
  { $match: { event_type: "LoadAttnV" } },
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
})' > load_attn_v.csv 

## Proj
mongosh --quiet attn_log --eval 'print("timestamp,counter,start,end"); db.log.aggregate([
  { $match: { event_type: "CompProj" } },
  { $project: { 
      _id: 0, 
      timestamp: 1, 
      counter: "$event_data.counter", 
      start: "$event_data.start", 
      end: "$event_data.end"
    }
  },
  { $sort: { timestamp: 1 } }
]).forEach(function(doc) { 
  print(doc.timestamp + "," + doc.counter + "," + doc.start + "," + doc.end); 
})' > comp_proj.csv 


mongosh --quiet attn_log --eval 'print("timestamp,counter,start,end"); db.log.aggregate([
  { $match: { event_type: "LoadProj" } },
  { $project: { 
      _id: 0, 
      timestamp: 1, 
      counter: "$event_data.counter", 
      start: "$event_data.start", 
      end: "$event_data.end"
    }
  },
  { $sort: { timestamp: 1 } }
]).forEach(function(doc) { 
  print(doc.timestamp + "," + doc.counter + "," + doc.start + "," + doc.end); 
})' > load_proj.csv 


mongosh --quiet attn_log --eval 'print("timestamp,counter,start,end"); db.log.aggregate([
  { $match: { event_type: "StoreProj" } },
  { $project: { 
      _id: 0, 
      timestamp: 1, 
      counter: "$event_data.counter", 
      start: "$event_data.start", 
      end: "$event_data.end"
    }
  },
  { $sort: { timestamp: 1 } }
]).forEach(function(doc) { 
  print(doc.timestamp + "," + doc.counter + "," + doc.start + "," + doc.end); 
})' > store_proj.csv 
