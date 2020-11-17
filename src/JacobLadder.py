import pylibxc
import re
from collections import Counter

def position(funct):

   iladder=1
   # get the all functionals in libxc
   xc_func_list = pylibxc.util.xc_available_functional_names()
   #print("xc_func_list",xc_func_list)

   # upper
   nfuncLIB=len(xc_func_list)
   for ifunc in range(nfuncLIB):
      xc_func_list[ifunc] = xc_func_list[ifunc].upper()

   # split the target functional, in case LC or CAM, etc.
   funct=funct.upper()
   funct_split=[]
   nsplit=len(funct.split("-"))
   for i in range(nsplit):
      funct_split.append(funct.split("-")[i]) 
   #print("Target functional : ",funct_split)   

   # --> find 
   xc_func_match=[]
   for ifunc in range(nfuncLIB):
      for i in range(nsplit):
         if xc_func_list[ifunc].find(funct_split[i]) != -1:
           xc_func_match.append(xc_func_list[ifunc])     

   if len(xc_func_match) == 0:
      ValueError: print("DFT functional isn't match that in libxc") 
#     print(xc_func_match)
#     print(Counter(xc_func_match))
   else:
       
      xc_func_match2 = [key for key,value in dict(Counter(xc_func_match)).items() if value > 1]
      #print(xc_func_match)

      # reduce the target region
      if len(xc_func_match2) == 0:
         xc_func_match2=xc_func_match 
          
      # --> re
      xc_func_exact=[]
      nmatch=len(xc_func_match2)
      for ifunc in range(nmatch):
         for i in range(nsplit):
            #print("RE.search ",re.search(funct_split[i],xc_func_match2[ifunc]),funct_split[i],xc_func_match2[ifunc]) 
            if re.search("_"+funct_split[i],xc_func_match2[ifunc]) != None:
               xc_func_exact.append(xc_func_match2[ifunc])  
            if re.search(funct_split[i]+"_",xc_func_match2[ifunc]) != None:
               xc_func_exact.append(xc_func_match2[ifunc])  

      #print(xc_func_exact)
      #print(Counter(xc_func_exact))
      xc_func_exact2 = [key for key,value in dict(Counter(xc_func_exact)).items()]

      #print("  ===>   Most possible functional in libxc is [",xc_func_exact2[0],"]")
      #print(xc_func_exact2[0].split("_"))

      if xc_func_exact2[0].split("_")[0] == "LDA":
         iladder = 1 
      elif xc_func_exact2[0].split("_")[0] == "GGA":
         iladder = 2 
      elif xc_func_exact2[0].split("_")[0] == "MGGA":
         iladder = 3 
      elif xc_func_exact2[0].split("_")[0] == "HYB":
         if xc_func_exact2[0].split("_")[1] == "LDA":
            iladder = 2
         elif xc_func_exact2[0].split("_")[1] == "GGA":
            iladder = 3
         elif xc_func_exact2[0].split("_")[1] == "MGGA":
            iladder = 4
         else:
            ValueError: print("Type of DFT functional isn't match  in libxc")  
      else:      
         ValueError: print("Type of DFT functional isn't match that in libxc")

   return iladder


def typicalfunct(iladder):

   if iladder==1:
      ref_funct="SVWN"  
   elif iladder==2:
      ref_funct="BLYP"  
   elif iladder==3:
      ref_funct="B3LYP"  
   elif iladder==4:
      ref_funct="M062X"  
   
   return ref_funct 

