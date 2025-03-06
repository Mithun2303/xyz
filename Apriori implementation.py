# -*- coding: utf-8 -*-
import copy
import pandas as pd

dataset = pd.read_csv("Groceries_dataset.csv")
dataset_g = {i[0]:list(i[1].itemDescription) for i in dataset.groupby(['Member_number','Date'])}
#dataset = dataset[0:100]



def find_support_count(support_i,transactions) :
  for i in support_i :
    for j in transactions.values() :
      #print(i,j
      if(set(i).issubset(set(j))) :
        support_i[i] += 1
  return support_i

def generate_next_itemset(itemsets) :
  next_itemsets = set()
  k = len(itemsets[0])+1
  for i in range(len(itemsets)) :
    for j in range(i+1,len(itemsets)) :
      item = itemsets[i].union(itemsets[j])
      if(len(item) == k) :
        next_itemsets.add(tuple(item))
  #next_itemsets = set(next_itemsets)
  return next_itemsets

def generate_subsets(itemsets) :
  k = len(itemsets)
  subsets = set()
  for i in range(k) :
    l = [itemsets[j] for j in range(k) if j!=i]
    subsets.add(tuple(l))
  return(subsets)

def prune(itemsets_1,itemsets_2) :
  support_2 = {}
  for i in itemsets_2 :
    f = 1
    subsets = generate_subsets(list(i))
    for j in subsets :
      if list(j) not in itemsets_1 :
        f=0
        break
    if f==1 :
      support_2[tuple(i)] = 0
  return support_2

def apriori(transactions,min_sup=10) :
  frequent_itemsets = {}
  items = set()
  for i in transactions.values() :
    for j in i :
      items.add(str(j))
  support_current = {tuple([i]):0 for i in items}
  support_current = find_support_count(support_current,transactions)
  itemsets_current = [set(i) for i in support_current if support_current[i]>=min_sup]

  l = 1
  while(True) :
    itemsets_next = generate_next_itemset(itemsets_current)
    itemsets_current = [list(i) for i in itemsets_current]
    #print("itemsets_current : ",itemsets_current)
    itemsets_next = [list(i) for i in itemsets_next]
    #print("itemsets_next : ",itemsets_next)
    support_next = prune(itemsets_current,itemsets_next)
    #print("support_next : ",support_next)
    support_next = find_support_count(support_next,transactions)
    #print("support_next : ",support_next)

    itemsets_next = [set(i) for i in support_next if support_next[i]>=min_sup]
    #print("itemsets_next : ",itemsets_next)

    if(len(itemsets_next)==1):
      print(l," : next")
      for i in support_next :
        if(support_next[i]>=min_sup):
          frequent_itemsets[i]=support_next[i]
      return frequent_itemsets
    elif(len(itemsets_next)==0):
      print(l," : current")
      f={tuple(i):0 for i in itemsets_current}
      f = find_support_count(f,transactions)
      t = {i:f[i] for i in f if f[i]>=min_sup}
      for i in t :
        frequent_itemsets[i] = t[i]
      return frequent_itemsets
    else :
      print("yes")
      # for i in support_next :
      #   if(support_next[i]>=min_sup):
      #     frequent_itemsets[i]=support_next[i]
      itemsets_current = copy.deepcopy(itemsets_next)
    l+=1

#min_sup = 100
frequent_itemsets = apriori(dataset_g,100)
for i in frequent_itemsets :
  print(i," : ",frequent_itemsets[i])

#min_sup = 50
frequent_itemsets = apriori(dataset_g,50)
for i in frequent_itemsets :
  print(i," : ",frequent_itemsets[i])

#min_sup = 10
frequent_itemsets = apriori(dataset_g,10)
for i in frequent_itemsets :
  print(i," : ",frequent_itemsets[i])

#min_sup = 500
frequent_itemsets = apriori(dataset_g,500)
for i in frequent_itemsets :
  print(i," : ",frequent_itemsets[i])

