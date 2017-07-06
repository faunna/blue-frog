admit.rows<-as.data.frame(lapply(as.data.frame.table(myTable.table),function(x)rep(x,as.data.frame.table(myTable.table)$Freq)))[,-freq-column]

