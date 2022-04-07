setwd("~/Desktop/IDAO 2022")
library(reshape2)
library(ggplot2)
library(spatialpred)
library(dengueThailand)
library(foreach)
library(ggplot2)
library(dplyr)
library(grid)
library(readr)
library(gridExtra)
library(cowplot)
library(reshape2)
library(rjags)
library(wesanderson)
library(cowplot)
library(scales)
library(pROC)
library(ResourceSelection)
library(readxl)
library(relaimpo)
library(ridge)
library(car)
library(mgcv)
library(glmnet)
library(caret)
library(tidyverse)

#read in data
targets=read.csv("~/Desktop/IDAO 2022/dichalcogenides_public/targets.csv")
targets$X_id=as.character(targets$X_id)

library(jsonlite)
setwd("~/Desktop/IDAO 2022/dichalcogenides_public/structures")

sampList=fromJSON(paste(targets$X_id[1], ".json", sep="") ) %>% as.list
#each json file for id is a list 
#list contains module, class, charge, lattice, sites
#module just says its pymatgen core structure
#class just says structur
#lattice gives coordinates 
#sites is dataframe
##abc gives coordinates



library("scatterplot3d") # load

x_cords=rep(0,length(sampList$sites$abc))
y_cords=rep(0,length(sampList$sites$abc))
z_cords=rep(0,length(sampList$sites$abc))
label=rep(0,length(sampList$sites$abc))

for(i in 1:length(sampList$sites$abc)){
  x_cords[i]=sampList$sites$abc[i][[1]][1]
  y_cords[i]=sampList$sites$abc[i][[1]][2]
  z_cords[i]=sampList$sites$abc[i][[1]][3]
  label[i]=sampList$sites$label[i]
}
label[label=="Mo"] <- "red"
label[label=="S"] <- "blue"

scatterplot3d(x_cords, y_cords, z_cords, color = label)

#function to plot abc coordinates by label 

plot_abc <- function(id){
  sampList=fromJSON(paste(targets$X_id[id], ".json", sep="") ) %>% as.list
  
  x_cords=rep(0,length(sampList$sites$abc))
  y_cords=rep(0,length(sampList$sites$abc))
  z_cords=rep(0,length(sampList$sites$abc))
  label=rep(0,length(sampList$sites$abc))
  
  for(i in 1:length(sampList$sites$abc)){
    x_cords[i]=sampList$sites$abc[i][[1]][1]
    y_cords[i]=sampList$sites$abc[i][[1]][2]
    z_cords[i]=sampList$sites$abc[i][[1]][3]
    label[i]=sampList$sites$label[i]
  }
  
  colors=rep(0, length(sampList$sites$abc))
  unique_elements=unique(sampList$sites$label)
  for(i in 1:length(unique_elements)){
    colors[label==unique_elements[i]] <- i
  }

  
  s3d <- (scatterplot3d(x_cords, y_cords, z_cords, color = colors))
  #legend(legend=levels(label))
  legend("bottomright", legend = unique(sampList$sites$label),
         col =  c(1, 2, 3), pch = 16)
  return(s3d)
}


#function to plot xyz coordinates by label 

plot_xyz <- function(id){
  sampList=fromJSON(paste(targets$X_id[id], ".json", sep="") ) %>% as.list
  
  x_cords=rep(0,length(sampList$sites$xyz))
  y_cords=rep(0,length(sampList$sites$xyz))
  z_cords=rep(0,length(sampList$sites$xyz))
  label=rep(0,length(sampList$sites$xyz))
  
  for(i in 1:length(sampList$sites$xyz)){
    x_cords[i]=sampList$sites$xyz[i][[1]][1]
    y_cords[i]=sampList$sites$xyz[i][[1]][2]
    z_cords[i]=sampList$sites$xyz[i][[1]][3]
    label[i]=sampList$sites$label[i]
  }
  
  colors=rep(0, length(sampList$sites$xyz))
  unique_elements=unique(sampList$sites$label)
  for(i in 1:length(unique_elements)){
    colors[label==unique_elements[i]] <- i
  }
  
  
  s3d <- (scatterplot3d(x_cords, y_cords, z_cords, color = colors))
  #legend(legend=levels(label))
  legend("bottomright", legend = unique(sampList$sites$label),
         col =  c(1, 2, 3), pch = 16)
  return(s3d) 
}

targets$a=0
targets$b=0
targets$c=0
targets$alpha=0
targets$beta=0
targets$gamma=0
targets$volume=0

for(id in 1:nrow(targets)){
  sampList=fromJSON(paste(targets$X_id[id], ".json", sep="") ) %>% as.list
  targets$a[id]=sampList$lattice$a
  targets$b[id]=sampList$lattice$b
  targets$c[id]=sampList$lattice$c
  targets$alpha[id]=sampList$lattice$alpha
  targets$beta[id]=sampList$lattice$beta
  targets$gamma[id]=sampList$lattice$gamma
  targets$volume[id]=sampList$lattice$volume
}

#everything has same lattice structure and volume
# paper says tmdcs have intrinsic defects
#including vacancies, 
#adatoms, 
#grain boundaries


#substitutional impurities
targets$num_substitution=0
for(id in 1:nrow(targets)){
  sampList=fromJSON(paste(targets$X_id[id], ".json", sep="") ) %>% as.list
  x=table(sampList$sites$label)
  x=as.data.frame(x)
  targets$num_substitution[id]=length(x$Var1)-2
}


ggplot(targets, aes(x=as.factor(num_substitution), y= band_gap, col=as.factor(num_substitution))) +
  geom_jitter()+
  xlab("Number of Substitutions")


#get vacancies 
targets$num_vacancies=0
total_points=8*8*3
for(id in 1:nrow(targets)){
  sampList=fromJSON(paste(targets$X_id[id], ".json", sep="") ) %>% as.list

  targets$num_vacancies[id]=total_points-nrow(sampList$sites)
}

#type of vacancies 


get_diff_vectors <- function(x, y) {
  count_x <- table(x)
  count_y <- table(y)
  same_counts <- match(names(count_y), names(count_x))
  count_x[same_counts] <- count_x[same_counts] - count_y
  as.numeric(rep(names(count_x), count_x))
}

targets$mo_vacancies=0
targets$s_vacancies=0
targets$s_bottom_vacancies=0
targets$s_top_vacancies=0

sampList=fromJSON(paste(targets$X_id[4], ".json", sep="") ) %>% as.list
good_lattice=sampList$sites$xyz
good_lattice_x=0
good_lattice_y=0
good_lattice_z=0
for(i in 1:length(good_lattice)){
  good_lattice_x[i]=good_lattice[[i]][1]
  good_lattice_y[i]=good_lattice[[i]][2]
  good_lattice_z[i]=good_lattice[[i]][3]
  
}

for(id in which(targets$num_vacancies > 0)){
  sampList=fromJSON(paste(targets$X_id[id], ".json", sep="") ) %>% as.list
  bad_lattice=sampList$sites$xyz
  bad_lattice_x=0
  bad_lattice_y=0
  bad_lattice_z=0
  for(i in 1:length(bad_lattice)){
    bad_lattice_x[i]=bad_lattice[[i]][1]
    bad_lattice_y[i]=bad_lattice[[i]][2]
    bad_lattice_z[i]=bad_lattice[[i]][3]
    
  }
  for(i in 1:targets$num_vacancies[id]){
    if(get_diff_vectors(good_lattice_z, bad_lattice_z)[i] < 3){
      targets$s_vacancies[id]=targets$s_vacancies[id]+1
      targets$s_bottom_vacancies[id]=targets$s_bottom_vacancies[id]+1
    }
    if(get_diff_vectors(good_lattice_z, bad_lattice_z)[i] >5){
      targets$s_vacancies[id]=targets$s_vacancies[id]+1
      targets$s_top_vacancies[id]=targets$s_top_vacancies[id]+1
      
    }
    if(get_diff_vectors(good_lattice_z, bad_lattice_z)[i] > 3 & get_diff_vectors(good_lattice_z, bad_lattice_z)[i] < 4  ){
      targets$mo_vacancies[id]=targets$mo_vacancies[id]+1
    }
  }
}



#plot substiutions vs band gap, colored by vacancies
ggplot(targets, aes(x=as.factor(num_substitution), y= band_gap, col=as.factor(num_vacancies))) +
  geom_jitter()+
  xlab("Number of Substitutions")+
  ylab("Band Gap")




#plot vacancies vs band gap, colored by substitutions
ggplot(targets, aes(x=as.factor(num_vacancies), y= band_gap, col=as.factor(num_substitution))) +
  geom_jitter()+
  xlab("Number of Vacancies")+
  ylab("Band Gap")



#investigating two subset 
two_vacs=subset(targets, targets$num_vacancies==2)
#plot substiutions vs band gap, colored by vacancies
ggplot(two_vacs, aes(x=as.factor(num_substitution), y= band_gap, col=as.factor(mo_vacancies))) +
  geom_jitter()+
  xlab("Number of Substitutions")+
  ylab("Band Gap")

ggplot(two_vacs, aes(x=as.factor(num_substitution), y= band_gap, col=as.factor(s_vacancies))) +
  geom_jitter()+
  xlab("Number of Substitutions")+
  ylab("Band Gap")

ggplot(two_vacs, aes(x=as.factor(num_substitution), y= band_gap, col=as.factor(s_top_vacancies))) +
  geom_jitter()+
  xlab("Number of Substitutions")+
  ylab("Band Gap")

##creating subsets
two_vac_low=subset(two_vacs, two_vacs$band_gap<0.6)
two_vac_mid=subset(two_vacs, two_vacs$band_gap>0.6 & two_vacs$band_gap<0.9)
two_vac_high=subset(two_vacs, two_vacs$band_gap>0.9)

#side by side observation
i=5
par(mfrow=c(1,3)) 
plot_xyz(which(targets$X_id==two_vac_low$X_id[i]))
plot_xyz(which(targets$X_id==two_vac_mid$X_id[i]))
plot_xyz(which(targets$X_id==two_vac_high$X_id[i]))




#build initial model
init_model <- function(subs, vacancies, mo_vacancies){
  if(vacancies==0){
    return(sample(targets$band_gap[which(targets$num_vacancies==0)], 1, replace=TRUE)
           + rnorm(1, 0, density(targets$band_gap[which(targets$num_vacancies==0)])$bw))
  }
  
  if(vacancies==1){
    if(subs==1){
      return(sample(targets$band_gap[which(targets$num_vacancies==1 & targets$num_substitution==1)], 1, replace=TRUE)
             + rnorm(1, 0, density(targets$band_gap[which(targets$num_vacancies==1 & targets$num_substitution==1)])$bw))
    }else if(subs==2){
      return(sample(targets$band_gap[which(targets$num_vacancies==1 & targets$num_substitution==2)], 1, replace=TRUE)
             + rnorm(1, 0, density(targets$band_gap[which(targets$num_vacancies==1 & targets$num_substitution==2)])$bw))
      
    }else{
      return(0)
    }
  }
  
  if(vacancies==2){
    if(mo_vacancies==1  ){
      return(sample(targets$band_gap[which(targets$num_vacancies==2 & targets$mo_vacancies==1)], 1, replace=TRUE)
             + rnorm(1, 0, density(targets$band_gap[which(targets$num_vacancies==2 & targets$mo_vacancies==1)])$bw))
    }else if(mo_vacancies==0){
      return(sample(targets$band_gap[which(targets$num_vacancies==2 & targets$mo_vacancies==0)], 1, replace=TRUE)
             + rnorm(1, 0, density(targets$band_gap[which(targets$num_vacancies==2 & targets$mo_vacancies==0)])$bw))
    }else{
      return(0)
    }
  }
  
  if(vacancies==3){
    return(sample(targets$band_gap[which(targets$num_vacancies==3)], 1, replace=TRUE)
           + rnorm(1, 0, density(targets$band_gap[which(targets$num_vacancies==3)])$bw))
    
  }
  if(vacancies > 3){
    return(0)
  }
}

init_model(targets$num_substitution[1000], targets$num_vacancies[1000], targets$mo_vacancies[1000])
targets$band_gap[1000]



#create result
what_to_test_on=read.csv("~/Desktop/IDAO 2022/dichalcogenides_private/structures/testing.txt")
what_to_test_on_id=what_to_test_on$id

result_df=data.frame("id"=rep(0, 2967),"predictions"=rep(0,2967) )
setwd("~/Desktop/IDAO 2022/dichalcogenides_private/structures")


for(i in 1:length(what_to_test_on_id)){
  #sampList=fromJSON(paste(what_to_test_on_id[i], ".json", sep="") ) %>% as.list
  result_df$id[i]=what_to_test_on_id[i]
  #result_df$predictions=init_model()
}

#substitutional impurities
result_df$num_substitution=0
for(id in 1:nrow(result_df)){
  sampList=fromJSON(paste(result_df$id[id], ".json", sep="") ) %>% as.list
  x=table(sampList$sites$label)
  x=as.data.frame(x)
  
  result_df$num_substitution[id]=length(x$Var1)-2
}


#get vacancies 
result_df$num_vacancies=0
total_points=8*8*3
for(id in 1:nrow(result_df)){
  sampList=fromJSON(paste(result_df$id[id], ".json", sep="") ) %>% as.list
  
  result_df$num_vacancies[id]=total_points-nrow(sampList$sites)
}

#get types 

result_df$mo_vacancies=0
result_df$s_vacancies=0
result_df$s_bottom_vacancies=0
result_df$s_top_vacancies=0

sampList=fromJSON(paste(result_df$id[3], ".json", sep="") ) %>% as.list
good_lattice=sampList$sites$xyz
good_lattice_x=0
good_lattice_y=0
good_lattice_z=0
for(i in 1:length(good_lattice)){
  good_lattice_x[i]=good_lattice[[i]][1]
  good_lattice_y[i]=good_lattice[[i]][2]
  good_lattice_z[i]=good_lattice[[i]][3]
  
}

for(id in which(result_df$num_vacancies > 0)){
  sampList=fromJSON(paste(result_df$id[id], ".json", sep="") ) %>% as.list
  bad_lattice=sampList$sites$xyz
  bad_lattice_x=0
  bad_lattice_y=0
  bad_lattice_z=0
  for(i in 1:length(bad_lattice)){
    bad_lattice_x[i]=bad_lattice[[i]][1]
    bad_lattice_y[i]=bad_lattice[[i]][2]
    bad_lattice_z[i]=bad_lattice[[i]][3]
    
  }
  for(i in 1:result_df$num_vacancies[id]){
    if(get_diff_vectors(good_lattice_z, bad_lattice_z)[i] < 3){
      result_df$s_vacancies[id]=result_df$s_vacancies[id]+1
      result_df$s_bottom_vacancies[id]=result_df$s_bottom_vacancies[id]+1
    }
    if(get_diff_vectors(good_lattice_z, bad_lattice_z)[i] >5){
      result_df$s_vacancies[id]=result_df$s_vacancies[id]+1
      result_df$s_top_vacancies[id]=result_df$s_top_vacancies[id]+1
      
    }
    if(get_diff_vectors(good_lattice_z, bad_lattice_z)[i] > 3 & get_diff_vectors(good_lattice_z, bad_lattice_z)[i] < 4  ){
      result_df$mo_vacancies[id]=result_df$mo_vacancies[id]+1
    }
  }
}

for(i in 1:length(what_to_test_on_id)){
  result_df$predictions[i]=init_model(result_df$num_substitution[i], result_df$num_vacancies[i], result_df$mo_vacancies[i] )
}

result_df=result_df[,1:2]

write.csv(result_df, "submission.csv")




