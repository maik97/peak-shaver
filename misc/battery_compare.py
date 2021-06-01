import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
from datetime import datetime

df = pd.read_csv("total_power.csv")#.rolling(3).mean().dropna()
#df = pd.read_csv('total_power.csv', index_col='SensorDateTime', parse_dates=True)
df["total_power"] = df["total_power"].rolling(3).mean()
df = df.dropna()
print(df)


df['SensorDateTime'] = pd.to_datetime(df['SensorDateTime'])



#df['Zeit2'] = df['Zeit'].dt.hour + df['Zeit'].dt.minute

#df['Uhr'] = df['Zeit'].dt.time

Uhr = df['SensorDateTime']
Verbrauch = df["total_power"]
Verbrauch_neu=list()
Verbrauch_zu=list()
Shave = list()
Menge_Liste=list()
Kosten = list()


#ShaveWert=30
Menge_o=0
Menge_u=0
Verlauf = list()
test=0
kEnergie=300
kLeistung=150
Energie_S=0
Energie_N=0
Energie=50
Leistung=25
Wasser=list()
LION=list()
SMS=list()
Liste=list()
print("Maximaler Verbrauch: ")

print(max(Verbrauch))


for ShaveWert in range(84,20,-1):
    Liste.append(ShaveWert)
    Energie=0
    Energie_N=0

    print("Max Peak" + str(ShaveWert))
    for x in range(2,len(df)-2):
        if Verbrauch[x]>test:
            test=Verbrauch[x]
        
        if Menge_o>Menge_u:
            Menge_u=Menge_o 
        if Verbrauch[x]<ShaveWert:
            #Menge_u=Menge_u+((ShaveWert-Verbrauch[x])*0.25)
            #Status1=0
            Menge_o=Menge_o-(ShaveWert-Verbrauch[x])
            if Menge_o<0:
                Menge_o=0
        if Menge_o>Menge_u:
            Menge_u=Menge_o 
        if Verbrauch[x]>ShaveWert:
            Menge_o=Menge_o+((Verbrauch[x]-ShaveWert)/12)

        if Menge_o>Menge_u:
            Menge_u=Menge_o            
    #print("Benötigte Energie:  " + str(Menge_u)+"kWh   Benötigte Leistung: " +str(84-ShaveWert)) 
    print("Menge_u: "+str(Menge_u)+" Maximaler Peak: "+str(test))
    Z_W=(102*(84-ShaveWert))-((1500*(84-ShaveWert)+15*Menge_u)/15)
    Z_L=(102*(84-ShaveWert))-((150*(84-ShaveWert)+300*Menge_u)/15)
    Z_S=(102*(84-ShaveWert))-((300*(84-ShaveWert)+1000*Menge_u)/20)

    Wasser.append(Z_W)
    LION.append(Z_L)
    SMS.append(Z_S)
    # print(102*(84-ShaveWert))
    # print((1500*(84-ShaveWert)+1500*Menge_u)/15)
    #print("Gewinn: "+str(z) +" mit ShaveWert: "+str(ShaveWert))
    # print("Menge: "+str(Menge_u))



print(Wasser)
print(LION)
print(SMS)   
print(Liste)

plt.rcParams['font.size'] = '16'
plt.plot(Liste,Wasser,color="red", label="Wasserstoff") 
plt.plot(Liste,LION,color="green", label="LION") 
plt.plot(Liste,SMS,color="blue", label="SMS") 
plt.legend(loc='upper right')
plt.gca().invert_xaxis()
plt.xlabel("maximale Leistungsentnahme (kW)",fontsize=16)
plt.ylabel("mögliche Erspanis pro Jahr (€)",fontsize=16)

plt.show()

#Wasserstoff

#Lion
#SMS

#print(Energie)  
# for x in range(0,8835):
    
#     if Verbrauch[x]>ShaveWert:
#         Menge_o=Menge_o+Verbrauch[x]-ShaveWert
#     else:
#         Menge_u=Menge_u+ShaveWert-Verbrauch[x]    

# print("Oben:")
# print(Menge_o)
# print("Unten:")
# print(Menge_u)
# for x in range(0,8835):
#     #print(str(x) + "THIS IS X")
#     Kosten.append((8835-x)*102)
#     ShaveWert=x
#     Verlauf.append(x)
#     for i in Verbrauch:
#         if i>ShaveWert:
#             Menge=Menge+((i-ShaveWert))*0.25
#     #print("Shavewert:" + str(ShaveWert) + "  " + str(Menge) + " kWh TOTAL") 
#     z=((8835-ShaveWert)*kLeistung+Menge*kEnergie)/15

#     Menge_Liste.append(z)
#     Menge=0   
# print(Menge_Liste)
# print(len(Verlauf))
# print(Verbrauch)
# plt.rcParams['font.size'] = '16'
# plt.title("Kosten und Ersparnisse für verschiedene maximale Leistungsentnahmen")
# plt.plot(Verlauf,Menge_Liste,color="red", label="Kosten pro Jahr")
# plt.plot(Verlauf,Kosten,color="green", label="Ersparnisse pro Jahr")
# plt.xlabel("Maximale Leistungsentnahme")
# plt.ylabel("Kosten/Ersparnisse")
# plt.legend(loc='upper right')


# plt.fill_between(Verlauf,Menge_Liste,Kosten, where=np.array(Menge_Liste)<=np.array(Kosten), color="green", interpolate=True, alpha=0.5)
# plt.fill_between(Verlauf,Menge_Liste,Kosten, where=np.array(Kosten)<=np.array(Menge_Liste), color="red", interpolate=True, alpha=0.5)


# plt.show()

#print(str(Menge*0.25) + " kWh TOTAL")

# for i in range(0,96):
	
# 	if Verbrauch[i]>ShaveWert:
# 		Verbrauch_neu.append(ShaveWert)
# 		Verbrauch_zu.append(ShaveWert)
# 	if Verbrauch[i]<ShaveWert:
# 		Verbrauch_neu.append(Verbrauch[i])
# 		Verbrauch_zu.append(Verbrauch[i])



# for i in range(50,0,-1):
#     print(i)
#     print(str(Menge*0.25) + " kWh")
    


#     if Verbrauch_zu[i]<ShaveWert:
#     	if (ShaveWert-Verbrauch_zu[i]<Menge):
#     		Menge=Menge-(ShaveWert-Verbrauch_zu[i])
#     		Verbrauch_zu[i]=ShaveWert
    		
#     	else:
#     		Verbrauch_zu[i]=Verbrauch_zu[i]+Menge
#     		Menge=0
#     		break
 	
    	 	





# #print(len(Verbrauch_neu))
# print(str(Menge*0.25) + " kWh")
# print(Verbrauch_neu)
# print(Verbrauch_zu)




# #Plot
# plt.rcParams['font.size'] = '16'
# fig, axs = plt.subplots(2, gridspec_kw={'hspace': 0.3})
# fig.suptitle("Vergleich des Energieverlaufs vor und nach dem Peak Shaving",fontsize=25)
# axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%m'))
# axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%m'))
# axs[0].set_ylim([0,50])
# axs[1].set_ylim([0,50])

# #axs[0].set_title("Verlauf vor dem Peak Shaving")
# #axs[1].set_title("Verlauf nach dem Peak Shaving")
# axs[0].set_ylabel("Energieverlauf in kW",fontsize=18)
# axs[0].set_xlabel("Zeit",fontsize=18)
# axs[1].set_ylabel("Energieverlauf in kW",fontsize=18)
# axs[1].set_xlabel("Zeit",fontsize=18)



# #Plot 1
# axs[0].plot(Uhr,Verbrauch)
# axs[0].plot(Uhr,Shave,color="black",linestyle="--")

# S = np.array(Shave)
# V = np.array(Verbrauch)

# #axs[0].fill_between(Uhr,Verbrauch,Shave, where=S>=V, color="green", interpolate=True)
# axs[0].fill_between(Uhr,Verbrauch,Shave, where=S<=V, color="red", interpolate=True, alpha=0.5)

# #Plot 2

# axs[1].plot(Uhr,Verbrauch_neu)
# axs[1].plot(Uhr,Verbrauch_zu,color="red",linestyle="--")
# axs[1].plot(Uhr,Shave,color="black",linestyle="--")
# axs[1].fill_between(Uhr,Verbrauch_neu,Verbrauch_zu, where=np.array(Verbrauch_neu)<=np.array(Verbrauch_zu), color="green", interpolate=True, alpha=0.5)



# plt.show()






