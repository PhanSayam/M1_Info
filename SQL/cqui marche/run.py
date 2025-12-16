# cli.py
import sqlite3
import re
from datetime import datetime

DB_PATH = "covoiturage.db"


# =========================
#  Utilitaires BD
# =========================


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def insert(conn, sql, params):
    cur = conn.execute(sql, params)
    conn.commit()
    return cur.lastrowid


# =========================
#  Sélection utilisateur
# =========================


# def ajouter_utilisateur(conn):
#     print("\n=== AJOUTER UN UTILISATEUR ===")
#     email = input("Email : ").strip()
#     mot_de_passe = input("Mot de passe (hash ou placeholder) : ").strip()
#     prenom = input("Prénom : ").strip()
#     nom = input("Nom : ").strip()
#     date_naissance = input("Date de naissance (YYYY-MM-DD) : ").strip()

#     if not (email and mot_de_passe and prenom and nom and date_naissance):
#         print("Tous les champs sont obligatoires.")
#         return
#     if "@" not in email:
#         print("Email invalide.")
#         return
#     try:
#         datetime.strptime(date_naissance, "%Y-%m-%d")
#     except ValueError:
#         print("Format de date invalide (attendu YYYY-MM-DD).")
#         return

#     print("Rôle de l'utilisateur :")
#     print("1 - Conducteur")
#     print("2 - Passager")
#     print("3 - Conducteur et passager")
#     role = input("Choix : ").strip()

#     if role not in {"1", "2", "3"}:
#         print("Choix invalide.")
#         return

#     try:
#         cur = conn.execute(
#             """
#             INSERT INTO utilisateur(email, mot_de_passe_hash, prenom, nom, date_naissance)
#             VALUES (?, ?, ?, ?, ?);
#             """,
#             (email, mot_de_passe, prenom, nom, date_naissance),
#         )
#         utilisateur_id = cur.lastrowid

#         if role in {"1", "3"}:
#             conn.execute(
#                 "INSERT INTO conducteur(utilisateur_id) VALUES (?);",
#                 (utilisateur_id,),
#             )
#         if role in {"2", "3"}:
#             conn.execute(
#                 "INSERT INTO passager(utilisateur_id) VALUES (?);",
#                 (utilisateur_id,),
#             )

#         conn.commit()
#         print(f" Utilisateur créé (id={utilisateur_id}).")
#     except sqlite3.IntegrityError as e:
#         conn.rollback()
#         print(f"Impossible d'ajouter l'utilisateur (doublon ?) : {e}")
#     except sqlite3.Error as e:
#         conn.rollback()
#         print(f"Erreur lors de l'ajout : {e}")

def ajouter_utilisateur(conn):
    print("\n=== AJOUTER UN UTILISATEUR ===")

    # 1. Email avec validation Regex
    while True:
        email = input("Email : ").strip()
        if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            break
        print("Email invalide (format attendu : nom@domaine.com).")

    # 2. Mot de passe avec règle de longueur
    while True:
        mot_de_passe = input("Mot de passe (min 8 caractères) : ").strip()
        if len(mot_de_passe) >= 8:
            break
        print("Mot de passe trop court.")

    # 3. Identité
    while True:
        prenom = input("Prénom : ").strip()
        nom = input("Nom : ").strip()
        if prenom and nom:
            break
        print("Le nom et le prénom ne peuvent pas être vides.")

    # 4. Date de naissance avec vérification de l'âge (Majeur)
    while True:
        date_str = input("Date de naissance (YYYY-MM-DD) : ").strip()
        try:
            date_naissance_dt = datetime.strptime(date_str, "%Y-%m-%d")
            # Vérification basique de la majorité (365.25 jours)
            age = (datetime.now() - date_naissance_dt).days / 365.25
            
            if date_naissance_dt > datetime.now():
                print("La date ne peut pas être dans le futur.")
            elif age < 18:
                print(f"Inscription refusée : vous devez être majeur (âge actuel : {int(age)} ans).")
                return  # On quitte la fonction car l'utilisateur ne peut pas s'inscrire
            else:
                date_naissance = date_str
                break
        except ValueError:
            print("Format de date invalide (attendu YYYY-MM-DD).")

    # 5. Rôle
    print("Rôle de l'utilisateur :")
    print("1 - Conducteur")
    print("2 - Passager")
    print("3 - Conducteur et passager")
    
    while True:
        role = input("Choix : ").strip()
        if role in {"1", "2", "3"}:
            break
        print("Choix invalide.")

    # 6. Insertion
    try:
        cur = conn.execute(
            """
            INSERT INTO utilisateur(email, mot_de_passe_hash, prenom, nom, date_naissance)
            VALUES (?, ?, ?, ?, ?);
            """,
            (email, mot_de_passe, prenom, nom, date_naissance),
        )
        utilisateur_id = cur.lastrowid

        if role in {"1", "3"}:
            conn.execute("INSERT INTO conducteur(utilisateur_id) VALUES (?);", (utilisateur_id,))
        if role in {"2", "3"}:
            conn.execute("INSERT INTO passager(utilisateur_id) VALUES (?);", (utilisateur_id,))

        conn.commit()
        print(f"Utilisateur {prenom} {nom} créé avec succès (ID={utilisateur_id}).")
        
    except sqlite3.IntegrityError:
        conn.rollback()
        print(f"Erreur : L'email '{email}' est déjà utilisé.")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Erreur technique : {e}")

def choisir_conducteur(conn):
    print("\n=== LISTE DES CONDUCTEURS ===")
    cur = conn.execute(
        """
        SELECT u.id, u.prenom, u.nom, u.email
        FROM utilisateur u
        JOIN conducteur c ON c.utilisateur_id = u.id
        ORDER BY u.id;
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("Aucun conducteur dans la base.")
        return None

    for r in rows:
        print(f"{r['id']}: {r['prenom']} {r['nom']} ({r['email']})")

    while True:
        try:
            choix = int(
                input("ID du conducteur pour se connecter (0 pour annuler) : ").strip()
            )
        except ValueError:
            print("Veuillez entrer un nombre.")
            continue
        if choix == 0:
            return None
        if any(r["id"] == choix for r in rows):
            return choix
        print("ID invalide, réessaie.")


def choisir_passager(conn):
    print("\n=== LISTE DES PASSAGERS ===")
    cur = conn.execute(
        """
        SELECT u.id, u.prenom, u.nom, u.email
        FROM utilisateur u
        JOIN passager p ON p.utilisateur_id = u.id
        ORDER BY u.id;
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("Aucun passager dans la base.")
        return None

    for r in rows:
        print(f"{r['id']}: {r['prenom']} {r['nom']} ({r['email']})")

    while True:
        try:
            choix = int(
                input("ID du passager pour se connecter (0 pour annuler) : ").strip()
            )
        except ValueError:
            print("Veuillez entrer un nombre.")
            continue
        if choix == 0:
            return None
        if any(r["id"] == choix for r in rows):
            return choix
        print("ID invalide, réessaie.")


# =========================
#  Fonctions vues CONDUCTEUR
# =========================

def verifier_documents_actifs(conn, conducteur_id):
    """Vérifie si le conducteur a un Permis et un CT valides et non expirés."""
    cur = conn.execute(
        """
        SELECT type, COUNT(*) as cnt
        FROM document
        WHERE conducteur_id = ? 
          AND statut = 'VALIDE' 
          AND (date_expiration IS NULL OR date_expiration > date('now'))
        GROUP BY type;
        """,
        (conducteur_id,)
    )
    docs = {row['type'] for row in cur.fetchall()}
    
    # Doit posséder les deux types
    manquants = []
    if "PERMIS" not in docs: manquants.append("Permis")
    if "CONTROLE_TECHNIQUE" not in docs: manquants.append("Contrôle Technique")
    
    if manquants:
        print(f"Publication refusée. Documents valides manquants : {', '.join(manquants)}")
        return False
    return True

def afficher_trajets_conducteur(conn, conducteur_id):
    cur = conn.execute(
        """
        SELECT id, depart, arrivee, date_heure, prix,
               nb_places_total, nb_places_disponibles, statut
        FROM trajet
        WHERE conducteur_id = ?
        ORDER BY date_heure DESC;
        """,
        (conducteur_id,),
    )
    rows = cur.fetchall()
    if not rows:
        print("\nAucun trajet pour ce conducteur.")
        return

    print("\n=== VOS TRAJETS ===")
    print(
        f"{'ID':<4} {'Départ':<15} {'Arrivée':<15} {'Date/heure':<20} "
        f"{'Prix':<6} {'Tot':<4} {'Disp':<4} {'Statut':<10}"
    )
    print("-" * 90)
    for r in rows:
        print(
            f"{r['id']:<4} {r['depart']:<15} {r['arrivee']:<15} {r['date_heure']:<20} "
            f"{r['prix']:<6} {r['nb_places_total']:<4} {r['nb_places_disponibles']:<4} {r['statut']:<10}"
        )

# def ajouter_trajet(conn, conducteur_id):
#     print("\n=== AJOUTER UN TRAJET ===")
#     depart = input("Ville de départ : ").strip()
#     arrivee = input("Ville d'arrivée : ").strip()
#     date_heure = input("Date/heure (ex: 2025-12-07 14:00) : ").strip()
#     try:
#         prix = float(input("Prix (€) : ").strip())
#         nb_places = int(input("Nombre de places totales : ").strip())
#     except ValueError:
#         print("Valeur numérique invalide.")
#         return
#     consignes = input("Consignes (optionnel) : ").strip() or None

#     if nb_places <= 0:
#         print("Le nombre de places doit être positif.")
#         return
#     if prix < 0:
#         print("Le prix ne peut pas être négatif.")
#         return
#     try:
#         # Valide le format sans changer le stockage
#         datetime.fromisoformat(date_heure)
#     except ValueError:
#         print("Format de date/heure invalide (attendu ISO 8601, ex: 2025-12-07 14:00).")
#         return

#     try:
#         trajet_id = insert(
#             conn,
#             """
#             INSERT INTO trajet(
#                 conducteur_id, depart, arrivee, date_heure,
#                 prix, nb_places_total, nb_places_disponibles,
#                 consignes, statut
#             )
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PUBLIE');
#             """,
#             (
#                 conducteur_id,
#                 depart,
#                 arrivee,
#                 date_heure,
#                 prix,
#                 nb_places,
#                 nb_places,
#                 consignes,
#             ),
#         )
#         print(f" Trajet ajouté avec succès (id={trajet_id}).")
#     except sqlite3.Error as e:
#         print(f"Erreur lors de l'ajout du trajet : {e}")

# def ajouter_trajet(conn, conducteur_id):
#     print("\n=== PROPOSER UN TRAJET ===")
    
#     # 1. Précondition : Documents (UC2)
#     if not verifier_documents_actifs(conn, conducteur_id):
#         return

#     # 2. Saisie
#     depart = input("Ville de départ : ").strip()
#     arrivee = input("Ville d'arrivée : ").strip()
#     date_str = input("Date/heure (YYYY-MM-DD HH:MM) : ").strip()
    
#     try:
#         # Validation date future (UC2)
#         dt_trajet = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
#         if dt_trajet <= datetime.now():
#             print("La date du trajet doit être dans le futur.")
#             return
#     except ValueError:
#         print("Format de date invalide (attendu YYYY-MM-DD HH:MM).")
#         return

#     try:
#         prix = float(input("Prix (€) : ").strip())
#         nb_places = int(input("Nombre de places totales : ").strip())
#         if nb_places < 1 or prix < 0: raise ValueError
#     except ValueError:
#         print("Valeurs numériques invalides.")
#         return
        
#     consignes = input("Consignes (optionnel) : ").strip() or None

#     # 3. Insertion Atomique (Trajet + Étapes)
#     try:
#         conn.execute("BEGIN IMMEDIATE;")
        
#         cur = conn.execute(
#             """
#             INSERT INTO trajet(
#                 conducteur_id, depart, arrivee, date_heure,
#                 prix, nb_places_total, nb_places_disponibles,
#                 consignes, statut
#             )
#             VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PUBLIE');
#             """,
#             (conducteur_id, depart, arrivee, date_str, prix, nb_places, nb_places, consignes)
#         )
#         trajet_id = cur.lastrowid
        
#         # UC2: Ajout des étapes
#         ordre = 1
#         print("Ajout d'étapes intermédiaires (Laissez vide pour finir).")
#         while True:
#             lieu = input(f"Étape n°{ordre} : ").strip()
#             if not lieu:
#                 break
#             conn.execute(
#                 "INSERT INTO etape(trajet_id, ordre, lieu) VALUES (?, ?, ?);",
#                 (trajet_id, ordre, lieu)
#             )
#             ordre += 1

#         conn.commit()
#         print(f"Trajet publié avec succès (ID: {trajet_id}) avec {ordre-1} étapes.")
        
#     except sqlite3.Error as e:
#         conn.rollback()
#         print(f"Erreur SQL : {e}")

def ajouter_trajet(conn, conducteur_id):
    print("\n=== PROPOSER UN TRAJET ===")
    
    # 1. Vérifs et Saisies (HORS TRANSACTION)
    if not verifier_documents_actifs(conn, conducteur_id):
        return

    depart = input("Ville de départ : ").strip()
    arrivee = input("Ville d'arrivée : ").strip()
    date_str = input("Date/heure (YYYY-MM-DD HH:MM) : ").strip()
    
    try:
        dt_trajet = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
        if dt_trajet <= datetime.now():
            print("La date du trajet doit être dans le futur.")
            return
    except ValueError:
        print("Format de date invalide.")
        return

    try:
        prix = float(input("Prix (€) : ").strip())
        nb_places = int(input("Nombre de places totales : ").strip())
        if nb_places < 1 or prix < 0: raise ValueError
    except ValueError:
        print("Valeurs numériques invalides.")
        return
        
    consignes = input("Consignes (optionnel) : ").strip() or None

    # Saisie des étapes en mémoire (pour ne pas bloquer la BD)
    etapes_list = []
    ordre = 1
    print("Ajout d'étapes intermédiaires (Entrée vide pour finir).")
    while True:
        lieu = input(f"Étape n°{ordre} : ").strip()
        if not lieu:
            break
        etapes_list.append((ordre, lieu))
        ordre += 1

    # 2. Écriture Atomique (RAPIDE)
    try:
        # On ouvre la transaction juste au moment d'écrire
        conn.execute("BEGIN IMMEDIATE;")
        
        cur = conn.execute(
            """
            INSERT INTO trajet(
                conducteur_id, depart, arrivee, date_heure,
                prix, nb_places_total, nb_places_disponibles,
                consignes, statut
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PUBLIE');
            """,
            (conducteur_id, depart, arrivee, date_str, prix, nb_places, nb_places, consignes)
        )
        trajet_id = cur.lastrowid
        
        # Insertion groupée des étapes
        for ordre_etape, lieu_etape in etapes_list:
            conn.execute(
                "INSERT INTO etape(trajet_id, ordre, lieu) VALUES (?, ?, ?);",
                (trajet_id, ordre_etape, lieu_etape)
            )

        conn.commit()
        print(f"Trajet publié avec succès (ID: {trajet_id}) avec {len(etapes_list)} étapes.")
        
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Erreur SQL : {e}")


def afficher_documents_conducteur(conn, conducteur_id):
    cur = conn.execute(
        """
        SELECT id, type, fichier_url, date_expiration, statut
        FROM document
        WHERE conducteur_id = ?
        ORDER BY type, id;
        """,
        (conducteur_id,),
    )
    rows = cur.fetchall()
    if not rows:
        print("\nAucun document pour ce conducteur.")
        return

    print("\n=== VOS DOCUMENTS ===")
    print(f"{'ID':<4} {'Type':<20} {'URL':<40} " f"{'Expiration':<12} {'Statut':<10}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['id']:<4} {r['type']:<20} {r['fichier_url']:<40} "
            f"{r['date_expiration'] or '-':<12} {r['statut']:<10}"
        )


def ajouter_document_conducteur(conn, conducteur_id):
    print("\n=== AJOUTER UN DOCUMENT ===")
    print("1 - Permis")
    print("2 - Contrôle technique")
    choix = input("Type de document : ").strip()

    if choix == "1":
        doc_type = "PERMIS"
    elif choix == "2":
        doc_type = "CONTROLE_TECHNIQUE"
    else:
        print("Choix invalide.")
        return

    fichier_url = input("URL du fichier (ex: https://...): ").strip()
    date_expiration_input = input("Date d'expiration (YYYY-MM-DD, vide si aucune) : ").strip()
    statut = (
        input("Statut (EN_COURS / VALIDE / REFUSE, défaut=EN_COURS) : ").strip().upper()
        or "EN_COURS"
    )

    if statut not in {"EN_COURS", "VALIDE", "REFUSE"}:
        print("Statut invalide.")
        return
    if not fichier_url:
        print("URL requise.")
        return
    date_expiration = None
    if date_expiration_input:
        try:
            datetime.strptime(date_expiration_input, "%Y-%m-%d")
            date_expiration = date_expiration_input
        except ValueError:
            print("Format de date invalide (YYYY-MM-DD).")
            return

    try:
        conn.execute("BEGIN IMMEDIATE;")
        cur = conn.execute(
            """
            INSERT INTO document(conducteur_id, fichier_url, date_expiration, statut, type)
            VALUES (?, ?, ?, ?, ?);
            """,
            (conducteur_id, fichier_url, date_expiration, statut, doc_type),
        )
        doc_id = cur.lastrowid

        if doc_type == "PERMIS":
            numero = input("Numéro de permis : ").strip()
            date_obtention = input("Date d'obtention (YYYY-MM-DD) : ").strip()
            if not (numero and date_obtention):
                print("Numéro et date d'obtention requis.")
                conn.rollback()
                return
            datetime.strptime(date_obtention, "%Y-%m-%d")
            conn.execute(
                "INSERT INTO permis(document_id, numero, date_obtention) VALUES (?, ?, ?);",
                (doc_id, numero, date_obtention),
            )
        else:
            immat = input("Numéro d'immatriculation : ").strip()
            date_controle = input("Date du contrôle (YYYY-MM-DD) : ").strip()
            if not (immat and date_controle):
                print("Immatriculation et date de contrôle requis.")
                conn.rollback()
                return
            datetime.strptime(date_controle, "%Y-%m-%d")
            conn.execute(
                """
                INSERT INTO controle_technique(document_id, numero_immatriculation, date_controle)
                VALUES (?, ?, ?);
                """,
                (doc_id, immat, date_controle),
            )

        conn.commit()
        print(f" Document {doc_type} ajouté avec succès (id={doc_id}).")
    except ValueError:
        conn.rollback()
        print("Format de date invalide.")
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Erreur lors de l'ajout du document : {e}")


# =========================
#  Fonctions vues PASSAGER
# =========================


def lister_trajets_publiés(conn):
    cur = conn.execute(
        """
        SELECT t.id, t.depart, t.arrivee, t.date_heure, t.prix,
               t.nb_places_disponibles,
               u.prenom AS conducteur_prenom,
               u.nom AS conducteur_nom
        FROM trajet t
        JOIN conducteur c ON c.utilisateur_id = t.conducteur_id
        JOIN utilisateur u ON u.id = c.utilisateur_id
        WHERE t.statut = 'PUBLIE'
        ORDER BY t.date_heure;
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("\nAucun trajet publié.")
        return

    print("\n=== TRAJETS PUBLIÉS ===")
    print(
        f"{'ID':<4} {'Départ':<15} {'Arrivée':<15} {'Date/heure':<20} "
        f"{'Prix':<6} {'Disp':<4} {'Conducteur':<20}"
    )
    print("-" * 100)
    for r in rows:
        conducteur_nom = f"{r['conducteur_prenom']} {r['conducteur_nom']}"
        print(
            f"{r['id']:<4} {r['depart']:<15} {r['arrivee']:<15} {r['date_heure']:<20} "
            f"{r['prix']:<6} {r['nb_places_disponibles']:<4} {conducteur_nom:<20}"
        )


# def reserver_trajet(conn, passager_id):
#     print("\n=== RÉSERVER UN TRAJET ===")
#     lister_trajets_publiés(conn)
#     try:
#         trajet_id = int(input("ID du trajet à réserver (0 pour annuler) : ").strip())
#     except ValueError:
#         print("ID invalide.")
#         return
#     if trajet_id == 0:
#         return

#     try:
#         nb_places = int(input("Nombre de places à réserver : ").strip())
#     except ValueError:
#         print("Nombre invalide.")
#         return
#     if nb_places <= 0:
#         print("Le nombre de places doit être positif.")
#         return

#     try:
#         conn.execute("BEGIN IMMEDIATE;")
#         cur = conn.execute(
#             "SELECT nb_places_disponibles FROM trajet WHERE id = ? AND statut = 'PUBLIE';",
#             (trajet_id,),
#         )
#         row = cur.fetchone()
#         if row is None:
#             conn.rollback()
#             print("Trajet introuvable ou non disponible.")
#             return

#         dispo = row["nb_places_disponibles"]
#         if nb_places > dispo:
#             conn.rollback()
#             print(f"Pas assez de places disponibles (restant : {dispo}).")
#             return

#         conn.execute(
#             """
#             INSERT INTO reservation(passager_id, trajet_id, nb_places, statut)
#             VALUES (?, ?, ?, 'ACTIVE');
#             """,
#             (passager_id, trajet_id, nb_places),
#         )

#         update_res = conn.execute(
#             """
#             UPDATE trajet
#             SET nb_places_disponibles = nb_places_disponibles - ?
#             WHERE id = ? AND nb_places_disponibles >= ?;
#             """,
#             (nb_places, trajet_id, nb_places),
#         )
#         if update_res.rowcount != 1:
#             conn.rollback()
#             print("Impossible de mettre à jour les places (concurrence ?).")
#             return

#         conn.commit()
#         print(" Réservation créée et places mises à jour.")
#     except sqlite3.Error as e:
#         conn.rollback()
#         print(f"Erreur lors de la réservation : {e}")

# def reserver_trajet(conn, passager_id):
#     print("\n=== RÉSERVER UN TRAJET ===")
#     lister_trajets_publiés(conn)
    
#     try:
#         trajet_id = int(input("ID du trajet (0 pour annuler) : ").strip())
#     except ValueError:
#         print("ID invalide.")
#         return
#     if trajet_id == 0:
#         return

#     try:
#         nb_places = int(input("Nombre de places à réserver : ").strip())
#     except ValueError:
#         print("Nombre invalide.")
#         return
#     if nb_places <= 0:
#         print("Le nombre de places doit être positif.")
#         return

#     try:
#         conn.execute("BEGIN IMMEDIATE;")
        
#         # 1. Vérifier disponibilité places
#         cur = conn.execute(
#             "SELECT nb_places_disponibles FROM trajet WHERE id = ? AND statut = 'PUBLIE';",
#             (trajet_id,),
#         )
#         row = cur.fetchone()
#         if row is None:
#             conn.rollback()
#             print("Trajet introuvable ou non disponible.")
#             return

#         dispo = row["nb_places_disponibles"]
#         if nb_places > dispo:
#             conn.rollback()
#             print(f"Pas assez de places disponibles (restant : {dispo}).")
#             return

#         # 2. Vérifier existence d'une réservation précédente (CORRECTION ICI)
#         cur = conn.execute(
#             "SELECT id, statut FROM reservation WHERE passager_id = ? AND trajet_id = ?",
#             (passager_id, trajet_id)
#         )
#         existing_resa = cur.fetchone()

#         if existing_resa:
#             # Cas A : Déjà réservé et actif
#             if existing_resa['statut'] == 'ACTIVE':
#                 conn.rollback()
#                 print("Vous avez déjà une réservation active sur ce trajet.")
#                 return
            
#             # Cas B : Réservation existante mais annulée -> On réactive (UPDATE)
#             conn.execute(
#                 """
#                 UPDATE reservation 
#                 SET statut = 'ACTIVE', nb_places = ? 
#                 WHERE id = ?;
#                 """,
#                 (nb_places, existing_resa['id'])
#             )
#             print("🔄 Ancienne réservation réactivée.")
            
#         else:
#             # Cas C : Nouvelle réservation (INSERT)
#             conn.execute(
#                 """
#                 INSERT INTO reservation(passager_id, trajet_id, nb_places, statut)
#                 VALUES (?, ?, ?, 'ACTIVE');
#                 """,
#                 (passager_id, trajet_id, nb_places),
#             )

#         # 3. Décrémenter le stock (Commun aux cas B et C)
#         update_res = conn.execute(
#             """
#             UPDATE trajet
#             SET nb_places_disponibles = nb_places_disponibles - ?
#             WHERE id = ? AND nb_places_disponibles >= ?;
#             """,
#             (nb_places, trajet_id, nb_places),
#         )
        
#         if update_res.rowcount != 1:
#             conn.rollback()
#             print("Erreur critique : Concurrence sur les places.")
#             return

#         conn.commit()
#         print("Réservation confirmée et places mises à jour.")
        
#     except sqlite3.Error as e:
#         conn.rollback()
#         print(f"Erreur lors de la réservation : {e}")
        
def reserver_trajet(conn, passager_id):
    print("\n=== RÉSERVER UN TRAJET ===")
    lister_trajets_publiés(conn)
    
    try:
        trajet_id = int(input("ID du trajet (0 pour annuler) : ").strip())
    except ValueError:
        print("ID invalide.")
        return
    if trajet_id == 0:
        return

    try:
        nb_places = int(input("Nombre de places à réserver : ").strip())
    except ValueError:
        print("Nombre invalide.")
        return
    if nb_places <= 0:
        print("Le nombre de places doit être positif.")
        return

    try:
        conn.execute("BEGIN IMMEDIATE;")
        
        # 1. Vérifier disponibilité places
        cur = conn.execute(
            "SELECT nb_places_disponibles FROM trajet WHERE id = ? AND statut = 'PUBLIE';",
            (trajet_id,),
        )
        row = cur.fetchone()
        if row is None:
            conn.rollback()
            print("Trajet introuvable ou non disponible.")
            return

        dispo = row["nb_places_disponibles"]
        if nb_places > dispo:
            conn.rollback()
            print(f"Pas assez de places disponibles (restant : {dispo}).")
            return

        # 2. Vérifier existence d'une réservation précédente
        cur = conn.execute(
            "SELECT id, statut FROM reservation WHERE passager_id = ? AND trajet_id = ?",
            (passager_id, trajet_id)
        )
        existing_resa = cur.fetchone()

        if existing_resa:
            # Cas A : Déjà réservé et actif -> ON BLOQUE
            if existing_resa['statut'] == 'ACTIVE':
                conn.rollback()
                print("Vous avez déjà une réservation active sur ce trajet.")
                return
            
            # Cas B : Réservation existante mais annulée -> ON RÉACTIVE (UPDATE)
            conn.execute(
                """
                UPDATE reservation 
                SET statut = 'ACTIVE', nb_places = ? 
                WHERE id = ?;
                """,
                (nb_places, existing_resa['id'])
            )
            print("🔄 Ancienne réservation réactivée.")
            
        else:
            # Cas C : Nouvelle réservation -> INSERT
            conn.execute(
                """
                INSERT INTO reservation(passager_id, trajet_id, nb_places, statut)
                VALUES (?, ?, ?, 'ACTIVE');
                """,
                (passager_id, trajet_id, nb_places),
            )

        # 3. Décrémenter le stock (Commun aux cas B et C)
        update_res = conn.execute(
            """
            UPDATE trajet
            SET nb_places_disponibles = nb_places_disponibles - ?
            WHERE id = ? AND nb_places_disponibles >= ?;
            """,
            (nb_places, trajet_id, nb_places),
        )
        
        if update_res.rowcount != 1:
            conn.rollback()
            print("Erreur critique : Plus de places disponibles au moment de la validation.")
            return

        conn.commit()
        print("Réservation confirmée !")
        
    except sqlite3.Error as e:
        conn.rollback()
        # Gestion propre du trigger conducteur
        if "INTERDIT" in str(e):
            print(f"Action refusée : Vous êtes le conducteur de ce trajet.")
        else:
            print(f"Erreur technique : {e}")


def voir_reservations_passager(conn, passager_id):
    cur = conn.execute(
        """
        SELECT r.id, r.statut, r.nb_places,
               t.id AS trajet_id, t.depart, t.arrivee, t.date_heure, t.prix
        FROM reservation r
        JOIN trajet t ON t.id = r.trajet_id
        WHERE r.passager_id = ?
        ORDER BY t.date_heure DESC;
        """,
        (passager_id,),
    )
    rows = cur.fetchall()
    if not rows:
        print("\nAucune réservation pour ce passager.")
        return

    print("\n=== VOS RÉSERVATIONS ===")
    print(
        f"{'ResID':<6} {'TrajetID':<8} {'Départ':<15} {'Arrivée':<15} "
        f"{'Date/heure':<20} {'Places':<6} {'Statut':<18}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r['id']:<6} {r['trajet_id']:<8} {r['depart']:<15} {r['arrivee']:<15} "
            f"{r['date_heure']:<20} {r['nb_places']:<6} {r['statut']:<18}"
        )
        
def annuler_reservation(conn, passager_id):
    print("\n=== ANNULER UNE RÉSERVATION ===")
    
    # UC4: Lister uniquement les réservations actives futures
    cur = conn.execute(
        """
        SELECT r.id, r.nb_places, t.depart, t.arrivee, t.date_heure 
        FROM reservation r
        JOIN trajet t ON t.id = r.trajet_id
        WHERE r.passager_id = ? 
          AND r.statut = 'ACTIVE'
          AND t.date_heure > datetime('now')
        """,
        (passager_id,)
    )
    rows = cur.fetchall()
    
    if not rows:
        print("Aucune réservation annulable (active et future).")
        return

    for r in rows:
        print(f"{r['id']}: {r['depart']} -> {r['arrivee']} ({r['date_heure']}) - {r['nb_places']} places")

    try:
        res_id = int(input("ID réservation à annuler (0 = retour) : ").strip())
        if res_id == 0: return
    except ValueError:
        return

    # Vérification d'appartenance
    target = next((r for r in rows if r['id'] == res_id), None)
    if not target:
        print("ID invalide.")
        return

    try:
        conn.execute("BEGIN IMMEDIATE;")
        
        # 1. Update statut réservation
        conn.execute(
            "UPDATE reservation SET statut = 'ANNULEE_PASSAGER' WHERE id = ?;", 
            (res_id,)
        )
        
        # 2. Restitution des places au trajet
        conn.execute(
            """
            UPDATE trajet 
            SET nb_places_disponibles = nb_places_disponibles + ?
            WHERE id = (SELECT trajet_id FROM reservation WHERE id = ?);
            """,
            (target['nb_places'], res_id)
        )
        
        conn.commit()
        print("Réservation annulée. Les places ont été libérées.")
        
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Erreur technique : {e}")


# def laisser_avis(conn, passager_id):
#     print("\n=== LAISSER UN AVIS ===")
#     # On propose les trajets pour lesquels ce passager a une réservation
#     cur = conn.execute(
#         """
#         SELECT DISTINCT t.id, t.depart, t.arrivee, t.date_heure
#         FROM reservation r
#         JOIN trajet t ON t.id = r.trajet_id
#         WHERE r.passager_id = ?;
#         """,
#         (passager_id,),
#     )
#     trajets = cur.fetchall()
#     if not trajets:
#         print("Vous n'avez aucune réservation, impossible de laisser un avis.")
#         return

#     print("Trajets sur lesquels vous pouvez laisser un avis :")
#     for t in trajets:
#         print(f"{t['id']}: {t['depart']} -> {t['arrivee']} ({t['date_heure']})")

#     try:
#         trajet_id = int(input("ID du trajet (0 pour annuler) : ").strip())
#     except ValueError:
#         print("ID invalide.")
#         return
#     if trajet_id == 0:
#         return

#     # Vérifie que ce trajet fait bien partie des trajets réservés
#     if not any(t["id"] == trajet_id for t in trajets):
#         print("Vous n'avez pas réservé ce trajet.")
#         return

#     try:
#         note = int(input("Note (1 à 5) : ").strip())
#         if note < 1 or note > 5:
#             print("Note invalide.")
#             return
#     except ValueError:
#         print("Note invalide.")
#         return

#     commentaire = input("Commentaire : ").strip()
#     anonyme_input = (
#         input("Voulez-vous être anonyme ? (o/n, défaut=n) : ").strip().lower()
#     )
#     anonyme = 1 if anonyme_input == "o" else 0

#     date_pub = datetime.now().isoformat(timespec="seconds")

#     try:
#         insert(
#             conn,
#             """
#             INSERT INTO avis(passager_id, trajet_id, note, commentaire, anonyme, date_publication)
#             VALUES (?, ?, ?, ?, ?, ?);
#             """,
#             (passager_id, trajet_id, note, commentaire, anonyme, date_pub),
#         )
#         print(" Avis ajouté.")
#     except sqlite3.Error as e:
#         print(f"Erreur lors de l'ajout de l'avis : {e}")

def laisser_avis(conn, passager_id):
    print("\n=== LAISSER UN AVIS ===")
    
    # 1. On filtre les statuts pour exclure les annulations
    # 2. On vérifie que la date du trajet est passée
    cur = conn.execute(
        """
        SELECT DISTINCT t.id, t.depart, t.arrivee, t.date_heure
        FROM reservation r
        JOIN trajet t ON t.id = r.trajet_id
        WHERE r.passager_id = ?
          AND r.statut NOT IN ('ANNULEE_PASSAGER', 'ANNULEE_CONDUCTEUR')
          AND t.date_heure < datetime('now');
        """,
        (passager_id,),
    )
    trajets = cur.fetchall()
    
    if not trajets:
        print("Aucun trajet terminé et non-annulé disponible pour laisser un avis.")
        return

    print("Trajets sur lesquels vous pouvez laisser un avis :")
    for t in trajets:
        print(f"{t['id']}: {t['depart']} -> {t['arrivee']} ({t['date_heure']})")

    try:
        trajet_id = int(input("ID du trajet (0 pour annuler) : ").strip())
    except ValueError:
        print("ID invalide.")
        return
    if trajet_id == 0:
        return

    # Vérifie que l'ID choisi fait partie de la liste filtrée ci-dessus
    if not any(t["id"] == trajet_id for t in trajets):
        print("Ce trajet n'est pas éligible (futur, annulé ou non réservé).")
        return

    try:
        note = int(input("Note (1 à 5) : ").strip())
        if note < 1 or note > 5:
            print("Note invalide.")
            return
    except ValueError:
        print("Note invalide.")
        return

    commentaire = input("Commentaire : ").strip()
    anonyme_input = (
        input("Voulez-vous être anonyme ? (o/n, défaut=n) : ").strip().lower()
    )
    anonyme = 1 if anonyme_input == "o" else 0

    date_pub = datetime.now().isoformat(timespec="seconds")

    try:
        insert(
            conn,
            """
            INSERT INTO avis(passager_id, trajet_id, note, commentaire, anonyme, date_publication)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (passager_id, trajet_id, note, commentaire, anonyme, date_pub),
        )
        print("Avis ajouté.")
    except sqlite3.IntegrityError:
        print("Vous avez déjà laissé un avis sur ce trajet.")
    except sqlite3.Error as e:
        print(f"Erreur lors de l'ajout de l'avis : {e}")

# =========================
#  Menus
# =========================


def menu_conducteur(conn, conducteur_id):
    # Récup info utilisateur
    cur = conn.execute(
        "SELECT prenom, nom FROM utilisateur WHERE id = ?;", (conducteur_id,)
    )
    u = cur.fetchone()
    nom_affiche = f"{u['prenom']} {u['nom']}" if u else f"id={conducteur_id}"

    while True:
        print(f"\n=== MENU CONDUCTEUR ({nom_affiche}) ===")
        print("1 - Voir mes trajets")
        print("2 - Ajouter un trajet")
        print("3 - Voir mes documents")
        print("4 - Ajouter un document (permis / CT)")
        print("9 - Changer d'utilisateur / de rôle")
        print("0 - Quitter l'application")
        choix = input("Ton choix : ").strip()

        if choix == "1":
            afficher_trajets_conducteur(conn, conducteur_id)
        elif choix == "2":
            ajouter_trajet(conn, conducteur_id)
        elif choix == "3":
            afficher_documents_conducteur(conn, conducteur_id)
        elif choix == "4":
            ajouter_document_conducteur(conn, conducteur_id)
        elif choix == "9":
            break
        elif choix == "0":
            return False  # pour quitter complètement
        else:
            print("Choix invalide.")
    return True  # on revient au menu rôle


# def menu_passager(conn, passager_id):
#     cur = conn.execute(
#         "SELECT prenom, nom FROM utilisateur WHERE id = ?;", (passager_id,)
#     )
#     u = cur.fetchone()
#     nom_affiche = f"{u['prenom']} {u['nom']}" if u else f"id={passager_id}"

#     while True:
#         print(f"\n=== MENU PASSAGER ({nom_affiche}) ===")
#         print("1 - Lister les trajets publiés")
#         print("2 - Réserver un trajet")
#         print("3 - Voir mes réservations")
#         print("4 - Laisser un avis sur un trajet réservé")
#         print("9 - Changer d'utilisateur / de rôle")
#         print("0 - Quitter l'application")
#         choix = input("Ton choix : ").strip()

#         if choix == "1":
#             lister_trajets_publiés(conn)
#         elif choix == "2":
#             reserver_trajet(conn, passager_id)
#         elif choix == "3":
#             voir_reservations_passager(conn, passager_id)
#         elif choix == "4":
#             laisser_avis(conn, passager_id)
#         elif choix == "9":
#             break
#         elif choix == "0":
#             return False
#         else:
#             print("Choix invalide.")
#     return True

def menu_passager(conn, passager_id):
    cur = conn.execute(
        "SELECT prenom, nom FROM utilisateur WHERE id = ?;", (passager_id,)
    )
    u = cur.fetchone()
    nom_affiche = f"{u['prenom']} {u['nom']}" if u else f"id={passager_id}"

    while True:
        print(f"\n=== MENU PASSAGER ({nom_affiche}) ===")
        print("1 - Lister les trajets publiés")
        print("2 - Réserver un trajet")
        print("3 - Voir mes réservations")
        print("4 - Annuler une réservation")
        print("5 - Laisser un avis sur un trajet réservé")
        print("9 - Changer d'utilisateur / de rôle")
        print("0 - Quitter l'application")
        choix = input("Ton choix : ").strip()

        if choix == "1":
            lister_trajets_publiés(conn)
        elif choix == "2":
            reserver_trajet(conn, passager_id)
        elif choix == "3":
            voir_reservations_passager(conn, passager_id)
        elif choix == "4":
            annuler_reservation(conn, passager_id)
        elif choix == "5":
            laisser_avis(conn, passager_id)
        elif choix == "9":
            break
        elif choix == "0":
            return False
        else:
            print("Choix invalide.")
    return True

def main():
    conn = get_conn()
    try:
        while True:
            print("\n============================")
            print("  APPLICATION COVOITURAGE")
            print("============================")
            print("1 - Se connecter comme CONDUCTEUR")
            print("2 - Se connecter comme PASSAGER")
            print("3 - Ajouter un nouvel utilisateur")
            print("0 - Quitter")
            choix = input("Ton choix : ").strip()

            if choix == "1":
                conducteur_id = choisir_conducteur(conn)
                if conducteur_id is None:
                    continue
                if not menu_conducteur(conn, conducteur_id):
                    break
            elif choix == "2":
                passager_id = choisir_passager(conn)
                if passager_id is None:
                    continue
                if not menu_passager(conn, passager_id):
                    break
            elif choix == "3":
                ajouter_utilisateur(conn)
            elif choix == "0":
                print("👋 Au revoir !")
                break
            else:
                print("Choix invalide, réessaie.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
