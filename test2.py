# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 18:16:11 2017

@author: Gavin
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:07:54 2016

@author: Timo
"""
from pylab import *
import cv2
import matplotlib.gridspec as gridspec

img = cv2.imread("C:/Users/Timo/Desktop/Fleskleur.jpg")
img = img[0:int(img.shape[0]*0.9),]
cv2.flip(img,0,img)
#img = cv2.resize(img,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
img_gray = cv2 . cvtColor( img , cv2.COLOR_BGR2GRAY)
img_edges = cv2 . Canny ( img , 50 , 300)
img_edges2 = cv2 . Canny ( img , 50 , 75)
cv2.namedWindow("beeld1",cv2.WINDOW_NORMAL)
cv2.namedWindow("beeld2",cv2.WINDOW_NORMAL)

foto = img_edges#[0:int(img_edges.shape[0]*0.9),]

#cv2.flip(foto,0,foto)

vertical= foto.shape[0]# vertical size
horizontal = foto.shape[1]#horizotal size

histoXas=array([],dtype=uint32)# Make array
#Find all horizontal values
for hor in range(horizontal):
    histoXas= np.append (histoXas,foto[:vertical,hor].sum())
x_values = array(np.arange (horizontal),dtype=uint32) # X values for plot

# Transpose a array
fotot=foto.T
showfoto = foto

histoYas=array([],dtype=uint32)# Make array


#Histo Yas berekenen
for ver in range(vertical):
    histoYas= np.append (histoYas,fotot[:horizontal,ver].sum())

y_values = array(np.arange (vertical),dtype=uint32) # Y values for plot

#liquid Level bepalen
level  = np.argmax(histoYas)#[::-1])

#stappen bepalen
step =int(level/20)
lagen = array(np.arange (0,level,step),dtype=uint32)

i = 0
#img fles maken
fles = np.zeros_like(img)
volumepix = 0

while (i+1 < len(lagen)):
    #laag uitknippen
    laag = img_edges2[lagen[i]:lagen[i+1],]
    #histo Xas van laag berekenen
    histoXaslaag=array([],dtype=uint32)
    for hor in range(horizontal):
        histoXaslaag= np.append (histoXaslaag,laag[:laag.shape[0],hor].sum())
    #Linker grens bepalen
    left = np.argmax(histoXaslaag[0:int(len(histoXaslaag)/2)])
    #Rechter grens bepalen
    right = np.argmax(histoXaslaag[int(len(histoXaslaag)/2):len(histoXaslaag)])+int(len(histoXaslaag)/2)
    #Radius bepalen
    radius = (right-left)/2
    #Volume bepalen, nu nog in pixels
    volumepix += np.pi*radius*radius*step
    #Lijnen rand tekenen
    cv2.line(foto,(left,lagen[i]),(left,lagen[i+1]),(255,0,0),5)
    cv2.line(foto,(right,lagen[i]),(right,lagen[i+1]),(255,0,0),5)
    #Fles uitknippen
    fles[lagen[i]:lagen[i+1],left:right] = img[lagen[i]:lagen[i+1],left:right]
    i+=1
#Lijn level tekenen
#foto flippen voor weergave
cv2.flip(foto,0,foto)
cv2.flip(fles,0,fles)
cv2.line(foto,(0,int(level*-1+vertical)),(horizontal,int(level*-1+vertical)),(255,255,255),10)

#foto's weergeven
while True:
    cv2.imshow("beeld1",fles)
    cv2.imshow("beeld2",img_edges)
    if(cv2.waitKey(10) == 27): # als Escape key (toetscode 27) wordt
        cv2.destroyAllWindows()
        break
    
#shit weergeven    
print (level)
figure()
gs = gridspec.GridSpec(2, 2, width_ratios=[1,1]) #define 4 graphics
cv2.flip(foto,0,foto)
#assign graphics
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
ax4 = plt.subplot(gs[3])

# draw graphic 1 (image)
ax1.imshow (foto, origin='centre')
ax1.set_title('Original')
ax1.set_ylim(ax1.get_ylim()[::1])
ax1.axis('on')

# draw graphic 2 (plot values along Y)
ax2.plot(histoYas,y_values[::-1] ,'g-', linewidth = 1.0)
ax2.xaxis.set_ticks_position("top") # X values on top
ax2.set_ylim(ax2.get_ylim()[::1]) # invert Y value
ax2.set_ylim(vertical,0)
# ax2.set_title('Y-axis distibution')
ax2.grid(True)

# draw graphic 3 (plot values along X)
ax3.plot(x_values,histoXas, 'r-', linewidth = 1.0)
ax3.xaxis.set_ticks_position("top") # X values on top
ax3.set_ylim(ax3.get_ylim()[::-1]) # invert Y value
ax3.set_xlim(0,horizontal)
# ax3.set_title('X-axis distibution').set_verticalalignment('bottom')
ax3.grid(True)


# draw graphic 4 (hisogram gray values from 0(black) to 255(white)
ax4.hist(img_gray.flatten(),255, normed=1)
ax4.set_ylim(0,0.10)
show()

# Show all plots
