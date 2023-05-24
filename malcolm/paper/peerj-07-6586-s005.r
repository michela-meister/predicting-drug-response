library("dplyr")
library("asbio")
library("pwr")

#Power test function:
powerZtest = function(alpha = 0.05, sigma, n, delta){
  zcr = qnorm(p = 1-alpha, mean = 0, sd = 1)
  s = sigma/sqrt(n)
  power = 1 - pnorm(q = zcr, mean = (delta/s), sd = 1)
  return(power)
}

#Specify the directory containing the below input file:
setwd("/.")

#Input data file:
data.in = read.csv("measurements_through_september_with_control_nondup.csv", head=T)

#Specify the format type of each column
data.in$Study    = as.character(data.in$Study)
data.in$Study_ID = as.character(data.in$Study_ID)
data.in$Group    = as.character(data.in$Group)
data.in$Drug     = as.character(data.in$Drug)
data.in$Control  = as.character(data.in$Control)
data.in$MID      = as.character(data.in$MID)
data.in$Day      = as.numeric(as.character(data.in$Day))
data.in$Size     = as.numeric(as.character(data.in$Size))

#Docetaxel study IDs
doc.study = c("TM00214","J000100674","TM00355","TM00832","TM00999",
              "J000079689","J000080739","J000099327","J000100675","J000101173",
              "J000102184","J000103634","J000103917","TM00103","TM00107","TM00186",
              "TM00192","TM00193","TM00202","TM00212","TM00219","TM00222","TM00226",
              "TM00233","TM00246","TM00253","TM00256","TM00298","TM00302","TM00335",
              "TM00386","TM00877","TM01079","TM01117","TM01273","TM01278","TM01563",
              "TM01039","TM00199","TM00387")

#Only use Docetaxel studies:
data.in = subset(data.in, data.in$Study %in% doc.study)

#NOTE: The below code reuses the Cisplatin analysis script although it investigates Docetaxel studies!

#Add Cisplatin (Docetaxel) identifier column
data.in$Cisplatin = as.numeric(grepl("Docetaxel", data.in$Drug))

#Subset the data to only contain control and Cisplatin (Docetaxel) data
data.sub.1 = subset(data.in, (data.in$Control == 1) | (data.in$Cisplatin == 1))

#Determine which studies have Cisplatin (Docetaxel)
agg.cis = as.data.frame(aggregate(data.sub.1$Cisplatin, by = list(data.sub.1$Study), FUN=max))

#Subset out any studies that do not have Cisplatin (Docetaxel)
agg.cis.0 = subset(agg.cis, agg.cis$x == 0)

#data.sub.2 contains all data on studies that have Cisplatin (Docetaxel) and control
data.sub.2 = subset(data.sub.1, !(data.sub.1$Study %in% as.character(agg.cis.0$Group.1)))

#Determine if any study has more than 1 Cisplatin (Docetaxel) group (be sure that each group is compared to control); this is rare but ok
agg.group = aggregate(data.sub.2$Drug, by = list(data.sub.2$Study), FUN=unique)

#Remove any groups that are combination therapy as listed in the pattern object below.
patterns = c("Docetaxel+", "Docetaxel +", "Docetaxel/")

#Remove combination therapies with Group name including "Docetaxel+"
data.sub.2.a = data.sub.2[grep("Docetaxel\\+", data.sub.2$Drug, invert=T),]

#Remove combination therapies with Group name including "Docetaxel +"
data.sub.2.b = data.sub.2.a[grep("Docetaxel \\+", data.sub.2.a$Drug, invert=T),]

#Remove combination therapies with Group name including "Docetaxel/"
data.sub.2.c = data.sub.2.b[grep("Docetaxel\\/", data.sub.2.b$Drug, invert=T),]

#Set data.sub.2 to object with combination therapies excluded.
data.sub.2.old = data.sub.2
data.sub.2 = data.sub.2.c

#For each study keep only five day range: 0, 7, 14, 21, and 28.

#Create object containing all datasets to be analyzed
study.uni.2 = unique(data.sub.2$Study)

#Begin resampling analysis loop

#For each of the studies with both Docetaxel and Control groups perform the following:
i = 1

for(i in c(1:40)){
  #print(i)
  
  #Subset the main dataset to contain one study's data
  study.sub.1 = subset(data.sub.2, data.sub.2$Study == study.uni.2[i])
  
  #Identify the days used in the study
  day.uni = unique(study.sub.1$Day)
  
  #For the vector of days, subset it to contain only days that correspond to the following conditions:
  
  #Day 0 (These values should only be -1, 0, or 1!)
  if(0 %in% day.uni){
    max.0 = 0
  }else if(length(day.uni[(day.uni >= -1)  & (day.uni <= 1)]) != 0){
    max.0 = max(day.uni[(day.uni >= -1) & (day.uni <= 1)])
  }else{
    max.0 = NA
  }
  
  #Day 7 (These values can range from 4 to 10!)
  if(7 %in% day.uni){
    max.7 = 7
  }else if(length(day.uni[(day.uni >= 4)  & (day.uni <= 10)]) != 0){
    max.7 = max(day.uni[(day.uni >= 4)  & (day.uni <= 10)])
  }else{
    max.7 = NA
  }
  
  #Day 14 (These values can range from 11 to 17!)
  if(14 %in% day.uni){
    max.14 = 14
  }else if(length(day.uni[(day.uni >= 11)  & (day.uni <= 17)]) != 0){
    max.14 = max(day.uni[(day.uni >= 11)  & (day.uni <= 17)])
  }else{
    max.14 = NA
  }
  
  #Day 21 (These values can range from 18 to 24!)
  if(21 %in% day.uni){
    max.21 = 21
  }else if(length(day.uni[(day.uni >= 18)  & (day.uni <= 24)]) != 0){
    max.21 = max(day.uni[(day.uni >= 18)  & (day.uni <= 24)])
  }else{
    max.21 = NA
  }

  #Day 28 (These values can range from 25 to 31!)
  if(28 %in% day.uni){
    max.28 = 28
  }else if(length(day.uni[(day.uni >= 25)  & (day.uni <= 31)]) != 0){
      max.28 = max(day.uni[(day.uni >= 25)  & (day.uni <= 31)])
  }else{
    max.28 = NA
  }
  
  #Obtain a vector of (non-NA) days to keep for the study
  max.all.pre = c(max.0, max.7, max.14, max.21, max.28)
  max.all = max.all.pre[!is.na(max.all.pre)]
  
  #Subset the study to contain only the days specified in "max.all"
  study.sub.2 = subset(study.sub.1, study.sub.1$Day %in% max.all)
  
  #Create a new column that renames the Day to its "max.all" Day (either 0, 7, 14, 21, or 28).
  #-This new day assignment is called "Day_new"
  study.sub.3  = mutate(study.sub.2, Day_new = ifelse(Day == max.0, 0, 
                                               ifelse(Day == max.7, 7,
                                               ifelse(Day == max.14, 14,
                                               ifelse(Day == max.21, 21,
                                               ifelse(Day == max.28, 28, NA))))))
  
  #Vector of days in order
  max.all.o = unique(study.sub.3$Day_new)[order(unique(study.sub.3$Day_new))]
  
  #Use resampling to calcuate TC, rate-based TC, and mRECIST for each Docetaxel (treated group) and Control groups for each day
  j = 1
  
  #Determine the groups in the study
  group.uni = unique(study.sub.3$Drug)
  cis.uni = group.uni[grepl("Docetaxel", group.uni)]
  con.uni = group.uni[!grepl("Docetaxel", group.uni)]
  
  #For each Docetaxel group perform the resampling data analyses
  for(j in 1:length(cis.uni)){
    cis.grp = cis.uni[j]
    con.grp = con.uni
    
    #Dataset containing only the Docetaxel group
    data.cis = subset(study.sub.3, study.sub.3$Drug == cis.grp)
    
    #Dataset containing only the Control group
    data.con = subset(study.sub.3, study.sub.3$Drug == con.grp)
    
    #Remove duplicates in data
    data.cis.old = data.cis
    data.con.old = data.con
    
    data.cis = data.cis[!duplicated(data.cis),]
    data.con = data.con[!duplicated(data.con),]
    
    #Perform resampling for each day
    k = 1
    
    #For each day perform the data format changes:
    for(k in 1:length(max.all.o)){
      #Subset the Docetaxel and Control data for only the day specificed (k)
      day.cis.1 = subset(data.cis, data.cis$Day_new == max.all.o[k])
      day.con.1 = subset(data.con, data.con$Day_new == max.all.o[k])
      
      #Subset the Docetaxel and Control data for the first day (0) to the last day specified (k)
      #-If this is the first iteration of days (k = 1), then these following datasets will be equivalent the above created
      day.cis.rate = subset(data.cis, data.cis$Day_new %in% max.all.o[1:k])
      day.con.rate = subset(data.con, data.con$Day_new %in% max.all.o[1:k])
      
      #For rate-based T/C, set volumes < 50 to 50, and then log transform the volumes.
      day.cis.rate$SizeOrig = day.cis.rate$Size
      day.cis.rate$Size[day.cis.rate$Size < 50] = 50
      day.cis.rate$logSize50 = log10(day.cis.rate$Size)
      
      day.con.rate$SizeOrig = day.con.rate$Size
      day.con.rate$Size[day.con.rate$Size < 50] = 50
      day.con.rate$logSize50 = log10(day.con.rate$Size)
  
      
      #Docetaxel data over weeks
      #-Ensure that data are not duplicated, and each sample is present across all weeks!
      p = 1
      p.count = 0
      data.id.uni = NULL
      cis.id.uni = unique(day.cis.rate$MID)
      
      for(p in 1:length(cis.id.uni)){
        data.id.sub = subset(day.cis.rate, day.cis.rate$MID == cis.id.uni[p])
        data.id.sub.uni = data.id.sub[!duplicated(data.id.sub),]
      
        if((nrow(data.id.sub.uni) == length(max.all.o[1:k])) & (length(unique(data.id.sub.uni$Day_new)) == length(max.all.o[1:k]))){
          p.count = 1 + p.count
        }
        
        if((p.count == 1) & (length(unique(data.id.sub.uni$Day_new)) == length(max.all.o[1:k]))){
          data.id.uni = data.id.sub.uni
        }
        
        if((p.count > 1) & (length(unique(data.id.sub.uni$Day_new)) == length(max.all.o[1:k]))){
          data.id.uni = rbind(data.id.uni, data.id.sub.uni)
        }
      }
      
      #Set data.id.uni to day.cis.rate
      day.cis.rate.old = day.cis.rate
      day.cis.rate = data.id.uni
      

      #Control data over weeks
      #-Ensure that data are not duplicated, and each sample is present across all weeks!
      p = 1
      p.count = 0
      data.id.uni = NULL
      con.id.uni = unique(day.con.rate$MID)
      
      for(p in 1:length(con.id.uni)){
        data.id.sub = subset(day.con.rate, day.con.rate$MID == con.id.uni[p])
        data.id.sub.uni = data.id.sub[!duplicated(data.id.sub),]
        
        if((nrow(data.id.sub.uni) == length(max.all.o[1:k])) & (length(unique(data.id.sub.uni$Day_new)) == length(max.all.o[1:k]))){
          p.count = 1 + p.count
        }
        
        if((p.count == 1) & (length(unique(data.id.sub.uni$Day_new)) == length(max.all.o[1:k]))){
          data.id.uni = data.id.sub.uni
        }
        
        if((p.count > 1) & (length(unique(data.id.sub.uni$Day_new)) == length(max.all.o[1:k]))){
          data.id.uni = rbind(data.id.uni, data.id.sub.uni)
        }
      }
      
      #Set the duplicate cleaned "data.id.uni" to "day.cis.rate"
      day.con.rate.old = day.con.rate
      day.con.rate = data.id.uni
      
      #Determine the number and which samples are not used across all weeks
      day.cis.1.c = unique(day.cis.rate$MID)
      day.con.1.c = unique(day.con.rate$MID)
      
      day.cis.1.o = unique(day.cis.rate.old$MID)
      day.con.1.o = unique(day.con.rate.old$MID)
      
      day.cis.diff = subset(day.cis.1.o, !(day.cis.1.o %in% day.cis.1.c))
      day.con.diff = subset(day.con.1.o, !(day.con.1.o %in% day.con.1.c))
      
      day.cis.diff.n = length(day.cis.diff)
      day.con.diff.n = length(day.con.diff)
      
      if(day.cis.diff.n != 0){
        day.cis.diff.s = paste(day.cis.diff, collapse=",")
      }else{
        day.cis.diff.s = NA
      }
      
      if(day.con.diff.n != 0){
        day.con.diff.s = paste(day.con.diff, collapse=",")
      }else{
        day.con.diff.s = NA
      }
      
      #Determine the number of replicates in the Docetaxel and Control groups  
      day.cis.1.n = length(unique(day.cis.rate$MID))
      day.con.1.n = length(unique(day.con.rate$MID))
      
      #If Docetaxel and Control groups do not have the same replicate size, 
      #then remove the most deviant sample(s) (according to day 0) until replicate size is equal.
      if(day.cis.1.n > day.con.1.n){
        #Determine the difference in replicate size between Docetaxel and Control groups
        samp.diff = day.cis.1.n - day.con.1.n
        
        #Subset the data to evaluate the variance in Size for day 0
        day.cis.rate.0 = subset(day.cis.rate, day.cis.rate$Day_new == 0)
        
        #Determine the difference Size from the mean Size for each sample, and rank the sample (higher difference equals lower rank)
        day.cis.rate.0$Diff = abs(day.cis.rate.0$SizeOrig - mean(day.cis.rate.0$SizeOrig))
        day.cis.rate.0$Rank = rank(-day.cis.rate.0$Diff)
        
        #Select a random set of sample(s) to exclude
        samp.x = sample(c(1:nrow(day.cis.rate.0)), samp.diff) 
        
        #Remove the most deviant samples until the replicate sizes are equal per sample group
        day.cis.rate.0 = subset(day.cis.rate.0, !(day.cis.rate.0$Rank %in% samp.x))
        
        #Use the filtered data that no longer contains the deviant samples.
        day.cis.rate = subset(day.cis.rate, day.cis.rate$MID %in% day.cis.rate.0$MID)
      }else if (day.cis.1.n < day.con.1.n){
        #Determine the difference in replicate size between Docetaxel and Control groups
        samp.diff = day.con.1.n - day.cis.1.n
        
        #Subset the data to evaluate the variance in Size for day 0
        day.con.rate.0 = subset(day.con.rate, day.con.rate$Day_new == 0)
        
        #Determine the difference Size from the mean Size for each sample, and rank the sample (higher difference equals lower rank)
        day.con.rate.0$Diff = abs(day.con.rate.0$SizeOrig - mean(day.con.rate.0$SizeOrig))
        day.con.rate.0$Rank = rank(-day.con.rate.0$Diff)
        
        #Select a random set of sample(s) to exclude
        samp.x = sample(c(1:nrow(day.con.rate.0)), samp.diff)
        
        #Remove the random set
        day.con.rate.0 = subset(day.con.rate.0, !(day.con.rate.0$Rank %in% samp.x))
        
        #Use the filtered data that no longer contains the deviant samples.
        day.con.rate = subset(day.con.rate, day.con.rate$MID %in% day.con.rate.0$MID)
      }
      
      #Idenitfy a number replicates to run during resampling
      min.rep = min(c(day.cis.1.n, day.con.1.n))
      
      if(min.rep > 10){
        rep.all = c(min.rep, 10:3)
      }else if(min.rep <= 10 & min.rep > 3){
        rep.all = c(min.rep:3)
      }else{
        rep.all = min.rep
      }
        
      m = 1
      
      #Perform resampling using the replicate sample sizes specified in "min.rep"
      for(m in 1:length(rep.all)){
        ##################
        #T/C and Volumes
        ##################
        #Calculate the ratio of the mean volume of treated tumors over the mean volume of control tumors.
        
        #Create a subset of the rate data that is for the specified day "k"
        #-This ensures that the same data (that has samples used across all weeks) is used for all analyses
        day.cis.rate.sub = subset(day.cis.rate, day.cis.rate$Day_new == max.all.o[k])
        day.con.rate.sub = subset(day.con.rate, day.con.rate$Day_new == max.all.o[k])
        
        #Store the input data to an object that can be saved
        data.in.df = rbind(day.cis.rate.sub, day.con.rate.sub)
        
        #Record the full population TC value, Docetaxel and Control volume mean and standard deviation.
        if(m == 1){
          valu.tc.f = mean(day.cis.rate.sub$SizeOrig)/mean(day.con.rate.sub$SizeOrig)
          
          #Determine the tumor progression category.
          if(valu.tc.f < 0.10){
            tc.cat = "Highly_Active"
          }else if((valu.tc.f >= 0.1) & (valu.tc.f <= 0.42)){
            tc.cat = "Active"
          }else if(valu.tc.f > 0.42){
            tc.cat = "Not_Active"
          }else{
            tc.cat = "Missing"
          }
          
          #Optional output:
          #mean.cis.ts = mean(day.cis.rate.sub$SizeOrig)
          #sdev.cis.ts = sd(day.cis.rate.sub$SizeOrig)
          #mean.con.ts = mean(day.con.rate.sub$SizeOrig)
          #sdev.con.ts = sd(day.con.rate.sub$SizeOrig)
          
          tc.per.05  = NA
          tc.per.10  = NA
          tc.cat.per = NA
          tc.pow.l.10 = NA
          tc.pow.g.10 = NA
          tc.pow.l.42 = NA
          tc.pow.g.42 = NA
        }else{
          #Determine the percent of subset population TC values that differ from the full population by 0.05 & 0.10.
          tc.val = replicate(1000, (mean(sample(day.cis.rate.sub$SizeOrig, rep.all[m], replace=T))/
                                    mean(sample(day.con.rate.sub$SizeOrig, rep.all[m], replace=T))))
          
          tc.per = tc.val - valu.tc.f 

          #Calculate the power estimates for greater and less than thresholds of .10 and .42
          #Standard deviation
          tc.sig = sd(tc.val)
          
          #Ha and Ho for the 0.10 and 0.42 tests
          tc.pow.ha.10 = valu.tc.f
          tc.pow.ho.10 = .10
          
          tc.pow.ha.42 = valu.tc.f
          tc.pow.ho.42 = .42
          
          #Calculate the effects divided by the standard deviation (a.k.a., d)
          tc.pow.d.10  = (tc.pow.ha.10 - tc.pow.ho.10)/tc.sig
          tc.pow.d.42  = (tc.pow.ha.42 - tc.pow.ho.42)/tc.sig
          
          #Power tests for 0.10 and 0.42 greater and less than scenarios
          tc.pow.l.10 = pwr.norm.test(d = tc.pow.d.10, n = rep.all[m], sig.level = 0.05, alternative = "less")$power
          tc.pow.g.10 = pwr.norm.test(d = tc.pow.d.10, n = rep.all[m], sig.level = 0.05, alternative = "greater")$power
          tc.pow.l.42 = pwr.norm.test(d = tc.pow.d.42, n = rep.all[m], sig.level = 0.05, alternative = "less")$power
          tc.pow.g.42 = pwr.norm.test(d = tc.pow.d.42, n = rep.all[m], sig.level = 0.05, alternative = "greater")$power
          
          tc.per.05 = (sum(abs(tc.per) > 0.05)/1000)*100
          tc.per.10 = (sum(abs(tc.per) > 0.10)/1000)*100
          
          #Determine how many resampling tumor categories differ from the original population's category.
          y = 1
          for(y in 1:length(tc.val)){
            if(tc.val[y] < 0.10){
              tc.cat.y = "Highly_Active"
            }else if((tc.val[y] >= 0.1) & (tc.val[y] <= 0.42)){
              tc.cat.y = "Active"
            }else if(tc.val[y] > 0.42){
              tc.cat.y = "Not_Active"
            }else{
              tc.cat.y = "Missing"
            }
            
            if(y == 1){
              tc.cat.all = tc.cat.y 
            }else{
              tc.cat.all = c(tc.cat.all, tc.cat.y)
            }
          
            tc.cat.per = (sum(tc.cat.all != tc.cat)/1000)*100
          }
        }

        
        #################
        #T-TEST: Volumes
        #Perform a 2-sample t-test that compares the mean value derived from the entire dataset (m == 1) versus the mean derived from the subset ("m" != 1)
        #-If m == 1, then this is the full sample, and store the slope values to compare to the subset.
        ################
        if(m == 1){
          #Set the complete data to the results from the full dataset
          vol.cis.full = day.cis.rate.sub$SizeOrig
          vol.cis.per  = NA
          
          vol.con.full = day.con.rate.sub$SizeOrig
          vol.con.per  = NA
          
        }else{
          #Perform a 1000 t-tests that determine whether the mean of subset is different than the mean of the full dataset
          #-This results in 1000 p-values
          if(sd(vol.cis.full) != 0){
            vol.cis.rep = replicate(1000, t.test(vol.cis.full, sample(day.cis.rate.sub$SizeOrig, rep.all[m], replace=T), alternative = "two.sided")$p.value)
            
            #Calculate the percentage of results where p-values indicate a statistical difference in means
            vol.cis.per = (sum(vol.cis.rep < 0.05)/1000)*100
            
            #Perform a 1000 t-tests that determine whether the mean of subset is different than the mean of the full dataset
            #-This results in 1000 p-values
            vol.con.rep = replicate(1000, t.test(vol.con.full, sample(day.con.rate.sub$SizeOrig, rep.all[m], replace=T), alternative = "two.sided")$p.value)
            
            #Calculate the percentage of results where p-values indicate a statistical difference in means
            vol.con.per = (sum(vol.con.rep < 0.05)/1000)*100
          }else{
            vol.cis.per = 0
            vol.con.per = 0
          }
        }
        
        ##################
        #rate-based T/C
        #-Use "rate" data object that contains multiple days
        ##################

        #Docetaxel data
        #-Linear model returns the slope associated with Day (uses original days)
        #-Uses original Days and takes mean of slope calculated for each sample
        if(m == 1){
          lm.cis.r <- function(){
            data.sub <- subset(day.cis.rate, day.cis.rate$MID %in% sample(unique(day.cis.rate$MID), rep.all[m], replace=F))
            data.sub.id = unique(data.sub$MID)
            
            z = 1
            for(z in 1:length(data.sub.id)){
              fit.sub = subset(data.sub, data.sub$MID == data.sub.id[z])
              
              fit.r   <- lm(logSize50 ~ Day, data = fit.sub)
              fit.out <- coef(fit.r)[2]
              
              if(z == 1){
                lm.mean <- fit.out
              }else{
                lm.mean <- c(lm.mean, fit.out)
              }
            }
            
            return(lm.mean)
          }
        }else{
          lm.cis.r <- function(){
            data.id = sample(unique(day.cis.rate$MID), rep.all[m], replace=T)
            
            y = 1
            
            for(y in 1:length(data.id)){
              fit.cis.data = subset(day.cis.rate, day.cis.rate$MID %in% data.id[y])
              
              if(y == 1){
                data.sub = fit.cis.data
              }else{
                data.sub = rbind(data.sub, fit.cis.data)
              }
            }
            
            day.uni.l = length(unique(data.sub$Day))
            
            z = 1
            for(z in 1:(nrow(data.sub)/day.uni.l)){
              fit.sub = data.sub[c((1 + (day.uni.l*(z-1))):(day.uni.l + (day.uni.l*(z-1)))),]
              
              fit.r   <- lm(logSize50 ~ Day, data = fit.sub)
              fit.out <- coef(fit.r)[2]
              
              if(z == 1){
                lm.mean <- fit.out
              }else{
                lm.mean <- c(lm.mean, fit.out)
              }
            }
            
            return(lm.mean)
          }
        }

        
        #Control data
        #-Linear model returns the slope associated with Day (uses original days)
        #-Uses original Days and takes mean of slope calculated for each sample
        if(m == 1){
          lm.con.r <- function(){
            data.sub <- subset(day.con.rate, day.con.rate$MID %in% sample(unique(day.con.rate$MID), rep.all[m], replace=F))
            data.sub.id = unique(data.sub$MID)
            
            z = 1
            for(z in 1:length(data.sub.id)){
              fit.sub = subset(data.sub, data.sub$MID == data.sub.id[z])
              
              fit.r   <- lm(logSize50 ~ Day, data = fit.sub)
              fit.out <- coef(fit.r)[2]
              
              if(z == 1){
                lm.mean <- fit.out
              }else{
                lm.mean <- c(lm.mean, fit.out)
              }
            }
            
            return(lm.mean)
          }
        }else{
          lm.con.r <- function(){
            data.id = sample(unique(day.con.rate$MID), rep.all[m], replace=T)
            
            y = 1
            
            for(y in 1:length(data.id)){
              fit.con.data = subset(day.con.rate, day.con.rate$MID %in% data.id[y])
              
              if(y == 1){
                data.sub = fit.con.data
              }else{
                data.sub = rbind(data.sub, fit.con.data)
              }
            }
            
            day.uni.l = length(unique(data.sub$Day))
            
            z = 1
            for(z in 1:(nrow(data.sub)/day.uni.l)){
              fit.sub = data.sub[c((1 + (day.uni.l*(z-1))):(day.uni.l + (day.uni.l*(z-1)))),]
              
              fit.r   <- lm(logSize50 ~ Day, data = fit.sub)
              fit.out <- coef(fit.r)[2]
              
              if(z == 1){
                lm.mean <- fit.out
              }else{
                lm.mean <- c(lm.mean, fit.out)
              }
            }
            
            return(lm.mean)
          }
        }
        
        
        ################
        #rate-based TC
        ################
        #If not working only with Day 0, then calculate the rate-based T/C using the above specified functions
        if(max(max.all[1:k]) > 3){
          
          if(m == 1){
            valu.rtc.f = 10^( (mean(as.numeric(lm.cis.r())) - mean(as.numeric(lm.con.r()))) * max(day.cis.rate$Day))
            
            #Determine the tumor progression category.
            if(valu.rtc.f < 0.10){
              rtc.cat = "Highly_Active"
            }else if((valu.rtc.f >= 0.1) & (valu.rtc.f <= 0.42)){
              rtc.cat = "Active"
            }else if(valu.rtc.f > 0.42){
              rtc.cat = "Not_Active"
            }else{
              rtc.cat = "Missing"
            }
            
            rtc.per.05 = NA
            rtc.per.10 = NA
            rtc.pow.l.10 = NA
            rtc.pow.g.10 = NA
            rtc.pow.l.42 = NA
            rtc.pow.g.42 = NA
          }else{
            #Determine the percent of subset population RTC values that differ from the full population by 0.05 & 0.10.
            rtc.val = replicate(1000, 10^( (mean(as.numeric(lm.cis.r())) - mean(as.numeric(lm.con.r()))) * max(day.cis.rate$Day)))
            
            rtc.per = rtc.val - valu.rtc.f
            
            #Calculate the power estimates for greater and less than thresholds of .10 and .42
            #Standard deviation
            rtc.sig = sd(rtc.val)
            
            #Ha and Ho for the 0.10 and 0.42 tests
            rtc.pow.ha.10 = valu.rtc.f
            rtc.pow.ho.10 = .10
            
            rtc.pow.ha.42 = valu.rtc.f
            rtc.pow.ho.42 = .42
            
            #Calculate the effects divided by the standard deviation (a.k.a., d)
            rtc.pow.d.10  = (rtc.pow.ha.10 - rtc.pow.ho.10)/rtc.sig
            rtc.pow.d.42  = (rtc.pow.ha.42 - rtc.pow.ho.42)/rtc.sig
            
            #Power tests for 0.10 and 0.42 greater and less than scenarios
            rtc.pow.l.10 = pwr.norm.test(d = rtc.pow.d.10, n = rep.all[m], sig.level = 0.05, alternative = "less")$power
            rtc.pow.g.10 = pwr.norm.test(d = rtc.pow.d.10, n = rep.all[m], sig.level = 0.05, alternative = "greater")$power
            rtc.pow.l.42 = pwr.norm.test(d = rtc.pow.d.42, n = rep.all[m], sig.level = 0.05, alternative = "less")$power
            rtc.pow.g.42 = pwr.norm.test(d = rtc.pow.d.42, n = rep.all[m], sig.level = 0.05, alternative = "greater")$power
            
            rtc.per.05 = (sum(abs(rtc.per) > 0.05)/1000)*100
            rtc.per.10 = (sum(abs(rtc.per) > 0.10)/1000)*100
            
            #Determine how many resampling tumor categories differ from the original population's category.
            y = 1
            for(y in 1:length(rtc.val)){
              if(rtc.val[y] < 0.10){
                rtc.cat.y = "Highly_Active"
              }else if((rtc.val[y] >= 0.1) & (rtc.val[y] <= 0.42)){
                rtc.cat.y = "Active"
              }else if(rtc.val[y] > 0.42){
                rtc.cat.y = "Not_Active"
              }else{
                rtc.cat.y = "Missing"
              }
              
              if(y == 1){
                rtc.cat.all = rtc.cat.y 
              }else{
                rtc.cat.all = c(rtc.cat.all, rtc.cat.y)
              }
              
              rtc.cat.per = (sum(rtc.cat.all != rtc.cat)/1000)*100
            }
          }
        }else{
          rtc.cat.per = NA
          
          valu.rtc.f = NA
          rtc.per.05 = NA
          rtc.per.10 = NA
          rtc.pow.l.10 = NA
          rtc.pow.g.10 = NA
          rtc.pow.l.42 = NA
          rtc.pow.g.42 = NA
        }
        
        
        
        ################
        #Change in volume (Volume at day X - Volume at day 0)/ Volume at day 0
        ################
        #If not working only with Day 0, then calculate the change in volume (with respect to day 0)
        if(max(max.all[1:k]) > 3){
          
          day.cis.rec.0 = subset(day.cis.rate, day.cis.rate$Day_new == 0)
          day.cis.rec.m = subset(day.cis.rate, day.cis.rate$Day_new == max(day.cis.rate$Day_new))
          
          day.con.rec.0 = subset(day.con.rate, day.con.rate$Day_new == 0)
          day.con.rec.m = subset(day.con.rate, day.con.rate$Day_new == max(day.con.rate$Day_new))
          
          if(m == 1){
            #Docetaxel
            valu.cisv.f = (mean(as.numeric(day.cis.rec.m$SizeOrig)) - mean(as.numeric(day.cis.rec.0$SizeOrig))) / mean(as.numeric(day.cis.rec.0$SizeOrig))
            
            #Determine the tumor progression category.
            if(valu.cisv.f <= -0.42){
              cisv.cat = "Highly_Active"
            }else if((valu.cisv.f <= 1.5) & (valu.cisv.f > -0.42)){
              cisv.cat = "Active"
            }else if(valu.cisv.f > 1.5){
              cisv.cat = "Not_Active"
            }else{
              cisv.cat = "Missing"
            }
            
            cisv.per.05 = NA
            cisv.per.10 = NA
            
            #Control
            valu.conv.f = (mean(as.numeric(day.con.rec.m$SizeOrig)) - mean(as.numeric(day.con.rec.0$SizeOrig))) / mean(as.numeric(day.con.rec.0$SizeOrig))
            
            #Determine the tumor progression category.
            if(valu.conv.f <= -0.42){
              conv.cat = "Highly_Active"
            }else if((valu.conv.f <= 1.5) & (valu.conv.f > -0.42)){
              conv.cat = "Active"
            }else if(valu.conv.f > 1.5){
              conv.cat = "Not_Active"
            }else{
              conv.cat = "Missing"
            }
            
            conv.per.05 = NA
            conv.per.10 = NA
          }else{
            #Docetaxel
            cisv.0.val = replicate(1000, (mean(sample(day.cis.rec.0$SizeOrig, rep.all[m], replace=T))))
            cisv.m.val = replicate(1000, (mean(sample(day.cis.rec.m$SizeOrig, rep.all[m], replace=T))))
            
            cisv.val = (cisv.m.val - cisv.0.val)/cisv.0.val
            
            cisv.per = cisv.val - valu.cisv.f
            
            cisv.per.05 = (sum(abs(cisv.per) > 0.05)/1000)*100
            cisv.per.10 = (sum(abs(cisv.per) > 0.10)/1000)*100
            
            #Determine how many resampling tumor categories differ from the original population's category.
            y = 1
            for(y in 1:length(cisv.val)){
              if(cisv.val[y] <= -0.42){
                cisv.cat.y = "Highly_Active"
              }else if((cisv.val[y] <= 1.5) & (cisv.val[y] > -0.42)){
                cisv.cat.y = "Active"
              }else if(cisv.val[y] > 1.5){
                cisv.cat.y = "Not_Active"
              }else{
                cisv.cat.y = "Missing"
              }
              
              if(y == 1){
                cisv.cat.all = cisv.cat.y 
              }else{
                cisv.cat.all = c(cisv.cat.all, cisv.cat.y)
              }
              
              cisv.cat.per = (sum(cisv.cat.all != cisv.cat)/1000)*100
            }
            
            #Control
            conv.0.val = replicate(1000, (mean(sample(day.con.rec.0$SizeOrig, rep.all[m], replace=T))))
            conv.m.val = replicate(1000, (mean(sample(day.con.rec.m$SizeOrig, rep.all[m], replace=T))))
            
            conv.val = (conv.m.val - conv.0.val)/conv.0.val
            
            conv.per = conv.val - valu.conv.f
            
            conv.per.05 = (sum(abs(conv.per) > 0.05)/1000)*100
            conv.per.10 = (sum(abs(conv.per) > 0.10)/1000)*100
            
            #Determine how many resampling tumor categories differ from the original population's category.
            y = 1
            for(y in 1:length(conv.val)){
              if(conv.val[y] <= -0.42){
                conv.cat.y = "Highly_Active"
              }else if((conv.val[y] <= 1.5) & (conv.val[y] > -0.42)){
                conv.cat.y = "Active"
              }else if(conv.val[y] > 1.5){
                conv.cat.y = "Not_Active"
              }else{
                conv.cat.y = "Missing"
              }
              
              if(y == 1){
                conv.cat.all = conv.cat.y 
              }else{
                conv.cat.all = c(conv.cat.all, conv.cat.y)
              }
              
              conv.cat.per = (sum(conv.cat.all != conv.cat)/1000)*100
            }
          }
        }else{
          #Docetaxel
          cisv.cat.per = NA
          valu.cisv.f = NA
          cisv.per.05 = NA
          cisv.per.10 = NA
          
          #Control
          conv.cat.per = NA
          valu.conv.f = NA
          conv.per.05 = NA
          conv.per.10 = NA
        }
        

        
        #################
        #T-TEST: Docetaxel Slope
        #Perform a 2-sample t-test that compares the mean value derived from the entire dataset (m == 1) versus the mean derived from the subset ("m" != 1)
        #-If m == 1, then this is the full sample, and store the slope values to compare to the subset.
        #################
        if((max(max.all[1:k]) > 3)){
          if(m == 1){
            #Set the complete data to the results from the full dataset
            sl.cis.f = lm.cis.r()
            sl.cis.per = NA
          }else{
            #Perform a 1000 t-tests that determine whether the mean of subset is different than the mean of the full dataset
            #-This results in 1000 p-values
            sl.cis.rep = replicate(1000, t.test(lm.cis.r(), sl.cis.f, alternative = "two.sided")$p.value)
          
            #Calculate the percentage of results where p-values indicate a statistical difference in means
            sl.cis.per = (sum(sl.cis.rep < 0.05)/1000)*100
          }
        }else{
          sl.cis.per = NA
        }

        
        #################
        #T-TEST: Control Slope
        #Perform a 2-sample t-test that compares the mean value derived from the entire dataset (m == 1) versus the mean derived from the subset ("m" != 1)
        #-If m == 1, then this is the full sample, and store the slope values to compare to the subset.
        #################
        if((max(max.all[1:k]) > 3)){
          if(m == 1){
            #Set the complete data to the results from the full dataset
            sl.con.f = lm.con.r()
            sl.con.per = NA
            
          }else{
            #Perform a 1000 t-tests that determine whether the mean of subset is different than the mean of the full dataset
            #-This results in 1000 p-values
            sl.con.rep = replicate(1000, t.test(lm.con.r(), sl.con.f, alternative = "two.sided")$p.value)
            
            #Calculate the percentage of results where p-values indicate a statistical difference in means
            sl.con.per = (sum(sl.con.rep < 0.05)/1000)*100
          }
        }else{
          sl.con.per = NA
        }

        
        ##################
        #RTV
        #-Use "rate" data object that contains multiple days (use day 0 and maximum day)
        ##################
        if(max(max.all.o[1:k]) > 3){
          day.cis.rec.0 = subset(day.cis.rate, day.cis.rate$Day_new == 0)
          day.cis.rec.m = subset(day.cis.rate, day.cis.rate$Day_new == max(day.cis.rate$Day_new))
            
          day.con.rec.0 = subset(day.con.rate, day.con.rate$Day_new == 0)
          day.con.rec.m = subset(day.con.rate, day.con.rate$Day_new == max(day.con.rate$Day_new))
        
          
          ##################
          #Docetaxel RTV
          ##################
          #-Docetaxel data
          #-Calculate RTV which is the ratio of the tumor size at maximum day over the tumor size at day 0
          #Do not use replacement if using full data set
          if(m == 1){
            dif.cis.rtv <- function(){
              #Obtain a vector of sample IDs for "m" samples
              cis.rtv.id <- sample(day.cis.rec.0$MID, rep.all[m], replace=F)
              
              #Obtain the day 0 and day "m" data for the samples obtained above
              cis.rtv.0.samp <- subset(day.cis.rec.0, day.cis.rec.0$MID %in% cis.rtv.id)
              cis.rtv.m.samp <- subset(day.cis.rec.m, day.cis.rec.m$MID %in% cis.rtv.id)
              
              #Obtain the ratio of the tumor size of day "m" over day 0
              rtv.cis <- cis.rtv.m.samp$SizeOrig/cis.rtv.0.samp$SizeOrig
              return(rtv.cis)
            }
          }else{
            dif.cis.rtv <- function(){
              #Obtain a vector of sample IDs for "m" samples
              cis.rtv.id <- sample(day.cis.rec.0$MID, rep.all[m], replace=T)
              
              y = 1
              
              for(y in 1:length(cis.rtv.id)){
                cis.rtv.0.data = subset(day.cis.rec.0, day.cis.rec.0$MID %in% cis.rtv.id[y])
                cis.rtv.m.data = subset(day.cis.rec.m, day.cis.rec.m$MID %in% cis.rtv.id[y])
                
                if(y == 1){
                  cis.rtv.0.samp = cis.rtv.0.data
                  cis.rtv.m.samp = cis.rtv.m.data
                }else{
                  cis.rtv.0.samp = rbind(cis.rtv.0.samp, cis.rtv.0.data)
                  cis.rtv.m.samp = rbind(cis.rtv.m.samp, cis.rtv.m.data)
                }
              }
              
              #Obtain the ratio of the tumor size of day "m" over day 0
              rtv.cis <- cis.rtv.m.samp$SizeOrig/cis.rtv.0.samp$SizeOrig
              return(rtv.cis)
            }
          }

          #################
          #T-TEST: Docetaxel RTV
          #Perform a 2-sample t-test that compares the mean value derived from the entire dataset (m == 1) versus the mean derived from the subset ("m" != 1)
          #-If m == 1, then this is the full sample, and store the RTV values to compare to the subset.
          #################
          if(m == 1){
            #Set the complete data to the results from the full dataset
            rtv.cis.f = dif.cis.rtv()
            rtv.cis.per = NA
            
          }else{
            #Perform a 1000 2-sample t-tests that determine whether the mean of subset is different than the mean of the full dataset
            #-This results in 1000 p-values
            rtv.cis.rep = replicate(1000, t.test(dif.cis.rtv(), rtv.cis.f, alternative = "two.sided")$p.value)
            
            #Calculate the percentage of results where p-values indicate a statistical difference in means
            rtv.cis.per = (sum(rtv.cis.rep < 0.05)/1000)*100
          }
          
          
          ##################
          #Control RTV
          ##################
          #-Calculate RTV which is the ratio of the tumor size at maximum day over the tumor size at day 0
          #Do not use replacement if using full data set
          if(m == 1){
            dif.con.rtv <- function(){
              #Obtain a vector of sample IDs for "m" samples
              con.rtv.id <- sample(day.con.rec.0$MID, rep.all[m], replace=F)
              
              #Obtain the day 0 and day "m" data for the samples obtained above
              con.rtv.0.samp <- subset(day.con.rec.0, day.con.rec.0$MID %in% con.rtv.id)
              con.rtv.m.samp <- subset(day.con.rec.m, day.con.rec.m$MID %in% con.rtv.id)
              
              #Obtain the ratio of the tumor size of day "m" over day 0
              rtv.con <- con.rtv.m.samp$SizeOrig/con.rtv.0.samp$SizeOrig
              return(rtv.con)
            }
          }else{
            dif.con.rtv <- function(){
              #Obtain a vector of sample IDs for "m" samples
              con.rtv.id <- sample(day.con.rec.0$MID, rep.all[m], replace=T)
              
              y = 1
              
              for(y in 1:length(con.rtv.id)){
                con.rtv.0.data = subset(day.con.rec.0, day.con.rec.0$MID %in% con.rtv.id[y])
                con.rtv.m.data = subset(day.con.rec.m, day.con.rec.m$MID %in% con.rtv.id[y])
                
                if(y == 1){
                  con.rtv.0.samp = con.rtv.0.data
                  con.rtv.m.samp = con.rtv.m.data
                }else{
                  con.rtv.0.samp = rbind(con.rtv.0.samp, con.rtv.0.data)
                  con.rtv.m.samp = rbind(con.rtv.m.samp, con.rtv.m.data)
                }
              }
              
              #Obtain the ratio of the tumor size of day "m" over day 0
              rtv.con <- con.rtv.m.samp$SizeOrig/con.rtv.0.samp$SizeOrig
              return(rtv.con)
            }
          }

          #################
          #T-TEST: Control RTV
          #Perform a 2-sample t-test that compares the mean value derived from the entire dataset (m == 1) versus the mean derived from the subset ("m" != 1)
          #-If m == 1, then this is the full sample, and store the RTV values to compare to the subset.
          #################
          if(m == 1){
            #Set the complete data to the results from the full dataset
            rtv.con.f = dif.con.rtv()
            rtv.con.per = NA
            
          }else{
            #Perform a 1000 t-tests that determine whether the mean of subset is different than the mean of the full dataset
            #-This results in 1000 p-values
            rtv.con.rep = replicate(1000, t.test(dif.con.rtv(), rtv.con.f, alternative = "two.sided")$p.value)
            
            #Calculate the percentage of results where p-values indicate a statistical difference in means
            rtv.con.per = (sum(rtv.con.rep < 0.05)/1000)*100
          }
          
          
          ##################
          #RTV Ratio
          ##################
          if(m == 1){
            #MEAN
            mean.rtv.ratio.f = mean(dif.cis.rtv())/mean(dif.con.rtv())
            
            #Determine the tumor progression category.
            if(mean.rtv.ratio.f < 0.10){
              rtv.mean.cat = "Highly_Active"
            }else if((mean.rtv.ratio.f >= 0.1) & (mean.rtv.ratio.f <= 0.42)){
              rtv.mean.cat = "Active"
            }else if(mean.rtv.ratio.f > 0.42){
              rtv.mean.cat = "Not_Active"
            }else{
              rtv.mean.cat = "Missing"
            }
            
            mean.rtv.per.05 = NA
            mean.rtv.per.10 = NA
            mean.rtv.pow.l.10 = NA
            mean.rtv.pow.g.10 = NA
            mean.rtv.pow.l.42 = NA
            mean.rtv.pow.g.42 = NA
            
            #MEDIAN
            med.rtv.ratio.f  = median(dif.cis.rtv())/median(dif.con.rtv())
          
            #Determine the tumor progression category.
            if(med.rtv.ratio.f < 0.10){
              rtv.med.cat = "Highly_Active"
            }else if((med.rtv.ratio.f >= 0.1) & (med.rtv.ratio.f <= 0.42)){
              rtv.med.cat = "Active"
            }else if(med.rtv.ratio.f > 0.42){
              rtv.med.cat = "Not_Active"
            }else{
              rtv.med.cat = "Missing"
            }
            
            med.rtv.per.05 = NA
            med.rtv.per.10 = NA
            med.rtv.pow.l.10 = NA
            med.rtv.pow.g.10 = NA
            med.rtv.pow.l.42 = NA
            med.rtv.pow.g.42 = NA
          }else{
            #MEAN
            mean.rtv.val = replicate(1000, mean(dif.cis.rtv())/mean(dif.con.rtv()))

            mean.rtv.per = mean.rtv.val - mean.rtv.ratio.f
            
            #Calculate the power estimates for greater and less than thresholds of .10 and .42
            #Standard deviation
            mean.rtv.sig = sd(mean.rtv.val)
            
            #Ha and Ho for the 0.10 and 0.42 tests
            mean.rtv.pow.ha.10 = mean.rtv.ratio.f
            mean.rtv.pow.ho.10 = .10
            
            mean.rtv.pow.ha.42 = mean.rtv.ratio.f
            mean.rtv.pow.ho.42 = .42
            
            #Calculate the effects divided by the standard deviation (a.k.a., d)
            mean.rtv.pow.d.10  = (mean.rtv.pow.ha.10 - mean.rtv.pow.ho.10)/mean.rtv.sig
            mean.rtv.pow.d.42  = (mean.rtv.pow.ha.42 - mean.rtv.pow.ho.42)/mean.rtv.sig
            
            #Power tests for 0.10 and 0.42 greater and less than scenarios
            mean.rtv.pow.l.10 = pwr.norm.test(d = mean.rtv.pow.d.10, n = rep.all[m], sig.level = 0.05, alternative = "less")$power
            mean.rtv.pow.g.10 = pwr.norm.test(d = mean.rtv.pow.d.10, n = rep.all[m], sig.level = 0.05, alternative = "greater")$power
            mean.rtv.pow.l.42 = pwr.norm.test(d = mean.rtv.pow.d.42, n = rep.all[m], sig.level = 0.05, alternative = "less")$power
            mean.rtv.pow.g.42 = pwr.norm.test(d = mean.rtv.pow.d.42, n = rep.all[m], sig.level = 0.05, alternative = "greater")$power

            mean.rtv.per.05 = (sum(abs(mean.rtv.per) > 0.05)/1000)*100
            mean.rtv.per.10 = (sum(abs(mean.rtv.per) > 0.10)/1000)*100            
            
            y = 1
            for(y in 1:length(mean.rtv.val)){
              if(mean.rtv.val[y] < 0.10){
                mean.rtv.cat.y = "Highly_Active"
              }else if((mean.rtv.val[y] >= 0.1) & (mean.rtv.val[y] <= 0.42)){
                mean.rtv.cat.y = "Active"
              }else if(mean.rtv.val[y] > 0.42){
                mean.rtv.cat.y = "Not_Active"
              }else{
                mean.rtv.cat.y = "Missing"
              }
              
              if(y == 1){
                mean.rtv.cat.all = mean.rtv.cat.y 
              }else{
                mean.rtv.cat.all = c(mean.rtv.cat.all, mean.rtv.cat.y)
              }
              
              mean.rtv.cat.per = (sum(mean.rtv.cat.all != rtv.mean.cat)/1000)*100
            }
            
            #MEDIAN
            med.rtv.val  = replicate(1000, median(dif.cis.rtv())/median(dif.con.rtv()))
            
            med.rtv.per  = med.rtv.val - med.rtv.ratio.f
            
            #Calculate the power estimates for greater and less than thresholds of .10 and .42
            #Standard deviation
            med.rtv.sig = sd(med.rtv.val)
            
            #Ha and Ho for the 0.10 and 0.42 tests
            med.rtv.pow.ha.10 = med.rtv.ratio.f
            med.rtv.pow.ho.10 = .10
            
            med.rtv.pow.ha.42 = med.rtv.ratio.f
            med.rtv.pow.ho.42 = .42
            
            #Calculate the effects divided by the standard deviation (a.k.a., d)
            med.rtv.pow.d.10  = (med.rtv.pow.ha.10 - med.rtv.pow.ho.10)/med.rtv.sig
            med.rtv.pow.d.42  = (med.rtv.pow.ha.42 - med.rtv.pow.ho.42)/med.rtv.sig
            
            #Power tests for 0.10 and 0.42 greater and less than scenarios
            med.rtv.pow.l.10 = pwr.norm.test(d = med.rtv.pow.d.10, n = rep.all[m], sig.level = 0.05, alternative = "less")$power
            med.rtv.pow.g.10 = pwr.norm.test(d = med.rtv.pow.d.10, n = rep.all[m], sig.level = 0.05, alternative = "greater")$power
            med.rtv.pow.l.42 = pwr.norm.test(d = med.rtv.pow.d.42, n = rep.all[m], sig.level = 0.05, alternative = "less")$power
            med.rtv.pow.g.42 = pwr.norm.test(d = med.rtv.pow.d.42, n = rep.all[m], sig.level = 0.05, alternative = "greater")$power
            
            med.rtv.per.05  = (sum(abs(med.rtv.per) > 0.05)/1000)*100
            med.rtv.per.10  = (sum(abs(med.rtv.per) > 0.10)/1000)*100
            
            y = 1
            for(y in 1:length(med.rtv.val)){
              if(med.rtv.val[y] < 0.10){
                med.rtv.cat.y = "Highly_Active"
              }else if((med.rtv.val[y] >= 0.1) & (med.rtv.val[y] <= 0.42)){
                med.rtv.cat.y = "Active"
              }else if(med.rtv.val[y] > 0.42){
                med.rtv.cat.y = "Not_Active"
              }else{
                med.rtv.cat.y = "Missing"
              }
              
              if(y == 1){
                med.rtv.cat.all = med.rtv.cat.y 
              }else{
                med.rtv.cat.all = c(med.rtv.cat.all, med.rtv.cat.y)
              }
              
              med.rtv.cat.per = (sum(med.rtv.cat.all != rtv.med.cat)/1000)*100
            }
            
          }
          
        }else{
          rtv.cis.per = NA
          rtv.con.per = NA
          
          mean.rtv.ratio.f = NA
          med.rtv.ratio.f  = NA
          
          mean.rtv.per.05 = NA
          mean.rtv.per.10 = NA

          med.rtv.per.05 = NA
          med.rtv.per.10 = NA
          
          mean.rtv.cat.per = NA
          med.rtv.cat.per  = NA

          mean.rtv.pow.l.10 = NA
          mean.rtv.pow.g.10 = NA
          mean.rtv.pow.l.42 = NA
          mean.rtv.pow.g.42 = NA
          
          med.rtv.pow.l.10 = NA
          med.rtv.pow.g.10 = NA
          med.rtv.pow.l.42 = NA
          med.rtv.pow.g.42 = NA
        }
        
        ##################
        #Time to tumor size 1000 (Use old day assignment)
        #-Use "rate" data object that contains maximum number of days with maximum number of reps
        ##################
        if((m == 1) & (k == length(max.all.o))){
          #Fit line to size measurements taken at Day time points
          #-Determine the Day at which the tumor should hit a size of 1,000
          cis.fit = lm(SizeOrig ~ Day, data = day.cis.rate)
          cis.int = coef(cis.fit)[1]
          cis.slp = coef(cis.fit)[2]
          cis.d1k = round((1000 - cis.int)/cis.slp, 0)
          
          #Fit line to size measurements taken at Day time points
          #-Determine the Day at which the tumor should hit a size of 1,000
          con.fit = lm(SizeOrig ~ Day, data = day.con.rate)
          con.int = coef(con.fit)[1]
          con.slp = coef(con.fit)[2]
          con.d1k = round((1000 - con.int)/con.slp, 0)
        }else{
          cis.d1k = NA
          con.d1k = NA
        }
        
        data.out = c(study.uni.2[i], cis.grp, max.all.o[k], rep.all[m], 
                     day.cis.diff.n, day.cis.diff.s, day.con.diff.n, day.con.diff.s,
                     cis.d1k, con.d1k, 
                     round(valu.tc.f, 4), tc.per.05, tc.per.10, tc.cat.per, tc.pow.l.10, tc.pow.g.10, tc.pow.l.42, tc.pow.g.42,
                     vol.cis.per, vol.con.per,
                     round(valu.rtc.f, 4), rtc.per.05, rtc.per.10, rtc.cat.per, rtc.pow.l.10, rtc.pow.g.10, rtc.pow.l.42, rtc.pow.g.42,
                     sl.cis.per, sl.con.per,
                     rtv.cis.per, rtv.con.per, 
                     round(mean.rtv.ratio.f, 4), round(med.rtv.ratio.f, 4),
                     mean.rtv.per.05, mean.rtv.per.10, mean.rtv.cat.per, mean.rtv.pow.l.10, mean.rtv.pow.g.10, mean.rtv.pow.l.42, mean.rtv.pow.g.42,
                     med.rtv.per.05, med.rtv.per.10, med.rtv.cat.per, med.rtv.pow.l.10, med.rtv.pow.g.10, med.rtv.pow.l.42, med.rtv.pow.g.42,
                     round(valu.cisv.f, 4), cisv.per.05, cisv.per.10, cisv.cat.per,
                     round(valu.conv.f, 4), conv.per.05, conv.per.10, conv.cat.per)  
        
        names(data.out) = c("Study", "doc_Group", "Day", "N",
                            "doc_omit_N", "doc_omit_Samples", "con_omit_N", "con_omit_Samples",
                            "doc_day_Size1K", "con_day_Size1K", 
                            "TC_full","TC_diff_05", "TC_diff_10", "TC_cat", "tc.pow.l.10", "tc.pow.g.10", "tc.pow.l.42", "tc.pow.g.42",
                            "Vol_doc_diff", "Vol_con_diff",
                            "RTC_full", "RTC_diff_05", "RTC_diff_10", "RTC_cat", "rtc.pow.l.10", "rtc.pow.g.10", "rtc.pow.l.42", "rtc.pow.g.42",
                            "Slope_doc_diff", "Slope_con_diff",
                            "RTV_doc_diff", "RTV_con_diff", 
                            "RTV_mean_full", "RTV_med_full",
                            "RTV_mean_diff_05", "RTV_mean_diff_10", "RTV_mean_cat", "mean.rtv.pow.l.10", "mean.rtv.pow.g.10", "mean.rtv.pow.l.42", "mean.rtv.pow.g.42",
                            "RTV_med_diff_05", "RTV_med_diff_10", "RTV_med_cat", "med.rtv.pow.l.10", "med.rtv.pow.g.10", "med.rtv.pow.l.42", "med.rtv.pow.g.42",
                            "DocV_full","DocV_diff_05", "DocV_diff_10", "DocV_cat",
                            "ConV_full","ConV_diff_05", "ConV_diff_10", "ConV_cat")
        
        #Create statistics output table
        if((i == 1) & (j == 1) & (k == 1) & (m == 1)){
          data.out.df = data.out
        }else{
          data.out.df = rbind(data.out.df, data.out)
        }
        
        #Save the input data used in the analyses.
        if((i == 1) & (j == 1) & (k == 1) & (m == 1)){
          data.in.df.all = data.in.df
        }else if(m == 1){
          data.in.df.all = rbind(data.in.df.all, data.in.df)
        }
      }
    }
  }
}

write.table(data.out.df, "Malcolm_resampling_docetaxel_vs_control_results_v5.txt", sep="\t", row=F, quote=F)
write.table(data.in.df.all, "Malcolm_resampling_docetaxel_vs_control_input_v5.txt", sep="\t", row=F, quote=F)
