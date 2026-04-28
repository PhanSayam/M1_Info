# ord(c)-97 
# chr(x+97)

def vigenere_chiffre(clair,k):
    dico = {"é":"e", "è":"e", "ê":"e", "ë":"e", "à":"a", "î":"i", "ï":"i", "ù":"u", "û":"u", "ç":"c"}
    
    temp = clair.lower().replace(" ","")
    mot = []
    n = len(k)
    
    for i in temp :
        if (i in dico):
            mot.append(ord(dico[i])-97)
        else :
            mot.append(ord(i)-97)
            
    for i in range(len(mot)):
        lettre = ord(k[i%n])-97
        mot[i]= chr((mot[i]+lettre)%26 +97)
        
    return "".join(mot)
    
    
vigenere_chiffre("Exemple de chiffrement","gauss")

def vigenere_dechiffre(cryptogramme,k):
    mot = []
    taille = len(cryptogramme)
    n = len(k)
    for i in range(taille):
        cle = ord(k[i%n].lower())-97
        temp = ((ord(cryptogramme[i])-97 )- cle)% 26 +97   
        mot.append(chr(temp))  
        
    return "".join(mot) 

vigenere_dechiffre("kxyehrexwunizxjkmyfl","gauss")

def vigenere(texte,k,mode):
    mot = []
    taille = len(texte)
    taille_cle = len(k)
    
    for i in range(taille):
        cle = ord(k[i %taille_cle].lower())-97
        ord_lettre = ord(texte[i].lower())-97
        temp = (ord_lettre + mode*cle) %26 +97   
        mot.append(chr(temp)) 

    return "".join(mot)
