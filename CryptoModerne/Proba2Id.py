def proba_2_id(texte):
    texte = texte.lower()
    N = len(texte)
    div = N*(N-1)
    dictionnaire = {}
    res = 0
    
    for i in texte : 
        if i not in dictionnaire :
            dictionnaire[i]=1
        else : 
            dictionnaire[i]+=1
        
    for j in dictionnaire:
        res += dictionnaire[j]*(dictionnaire[j]-1)
        
    return res / div

# print(proba_2_id('Bonjour a tous, le soleil brille les etoiles aussi, What a wonderfull world !!!'))
     	

def extract(texte,n):
    return texte[0:len(texte):n]


def longueur_cle(crypto):
    i = 1
    texte = extract(crypto,i)
    Ic_T = proba_2_id(texte)
    while(Ic_T < 0.065):
        texte = extract(crypto,i)
        Ic_T = proba_2_id(texte)
        i+=1
    return i-1

def cesar(cryptogramme):
    dico = {}
    for i in cryptogramme : 
        if i not in dico :
            dico[ord(i)]=1
        else : 
            dico[ord(i)]+=1
            
    cle_val_max = max(dico, key=dico.get)
    decalage = cle_val_max - 32 % 95
    
    return decalage
    

def Icm(texte, sous_texte, i):
    texte = texte.lower()
    sous_texte = sous_texte.lower()
    dico_texte = {}
    n = len(texte)
    
    dico_sous_texte = {}
    m = len(sous_texte)
    
    for i in texte : 
        if i not in dico_texte :
            dico_texte[ord(i)]=1
        else : 
            dico_texte[ord(i)]+=1
            
    for i in sous_texte : 
        if i not in dico_sous_texte :
            dico_sous_texte[ord(i)]=1
        else : 
            dico_sous_texte[ord(i)]+=1
            
    print(dico_texte)
    print(dico_sous_texte)
    
    
           

cryptogramme = "30556e5b4f525662744f5a4e5d586f52534f524b5d69694b5b734755635b4f64505e6a5c5452596354754c5e4a716e5a5e655955635e505b5f574b6e68704b63594d5c5b4853596553645f495261495e63664b5a65616e5e4e5a596354545a695747686458596c58596f6256564a4f5b2369325267635161495c4a596e6b4f5851635d515d5b5d6459755a5e4b5b58455954555e64476e654f58615963545466605a4b7a74594c5366455e63665f536562654f496155475c54664b554758684b54617349646e4c585b5564694b546122633164664e4a5a5e6a5c65517a595e6e5a4f535a585a5c716c5d506f614c5848555d695c476c69526f65504f4e526e5d59535a596361645069584b5c5756475668636054594e5a65535658596c6749636e574f5359545a5d736c3d506f5b6d4f535a615653544e73456f624869585b58694f716c6059596e4d4b4e5950635e655159475f645d5c4e586e614f596c68565562565c586552564d4e52676354546656466555645c4b612263395b5a69574b5d5859546166496254555e644a546869475b5d5151645f694a5a5e6358475b68577b6e4a59525354744e4b6073495364594f5a4f5b68694b60644d5556534f58655e6a694a5267636254554b574a62745c5b6059577d6e334f645c585a53526c5c535d5c4c69514b627459486059566650505e6447655a4d6552614962654c53515254624f5461206360614c5846546374564b6c68495d5f5a69496c50655a5852574d5561664d4d47606a4f655662576450555e645650685d4b6c59526f5b4c5f5765526457564e5b52595474692e5262744b5a61594d575d505c4a5463744c4f5262585f63665f534b6e58564756664d55614c69545b6e6a58655a554b5e584d53565b54744d4b5f5a635c545a69465a635a584a4e5d587b6e544b4f4b62695f4b626c635563665352565e684b546122633c5466604e4b5861694e5c6151556e5a704a546e565a565f63475850664e545b525a574b5b686f6f5b4c5d645f546a62654f664d5c5b485859596e59704b5a6358595e5577642f5b745d4b5a5650516e5a704a5463674f5a52624d626e48604a496e6170475b5d51515b726948555c624f65607a4d5c62665d4a65526457565f595251584c5859655f565c4c4e5d58555c4c5859736e455f4f602063515f594f5865606a4f525e6949636e5058585a50635e59787350556e4a4f574c6e68704b59634d575d4869514b5d694f535262587b6e534b4e596256585a6c60496f65504f4e526e5d59535a59635450555d645b5d5a69495c6258555c5756465a586458655d554d635849564a736e414f655759595e54665254535c5a69495c615462585b6946525e675d655e69496f524c5e594b6e674f5450635264614c69465c505e5e655268496f64556952555c5a585a6c695259605c4f644b63745a5852574d55645f756457647b53526c626a5f6449564e4b6156535a6c5e455d50505d72"
cryptogramme = bytes.fromhex(cryptogramme).decode('utf-8')

taille_cle = longueur_cle(cryptogramme)
print(taille_cle)

sous_texte = extract(cryptogramme,taille_cle)



Icm(cryptogramme,sous_texte, taille_cle)
