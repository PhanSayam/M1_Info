# insert_test_data.py
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta

DB_PATH = "covoiturage.db"

def hacher_mdp(mot_de_passe):
    """Génère un sel et hashe le mot de passe (compatible avec la vérification du CLI)."""
    salt = secrets.token_hex(16)
    hash_bytes = hashlib.pbkdf2_hmac(
        'sha256', 
        mot_de_passe.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000
    )
    return f"{salt}:{hash_bytes.hex()}"

def insert(conn, sql, params):
    """Exécute un INSERT et retourne lastrowid."""
    cur = conn.execute(sql, params)
    return cur.lastrowid


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")

    print("Insertion de données de test...")

    # =========================================================
    # 1. Utilisateurs : un conducteur et un passager
    # =========================================================

    # Conducteur (Alice) - Mot de passe : "secret"
    conducteur_id = insert(
        conn,
        """
        INSERT INTO utilisateur(email, mot_de_passe_hash, prenom, nom, date_naissance)
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            "conducteur@example.com",
            hacher_mdp("secret"),  # <--- CORRIGÉ (Hashage réel)
            "Alice",
            "Dupont",
            "1990-05-10",
        ),
    )

    # Passager (Bob) - Mot de passe : "secret"
    passager_id = insert(
        conn,
        """
        INSERT INTO utilisateur(email, mot_de_passe_hash, prenom, nom, date_naissance)
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            "passager@example.com",
            hacher_mdp("secret"),  # <--- CORRIGÉ (Hashage réel)
            "Bob",
            "Martin",
            "1995-09-22",
        ),
    )

    # Tables filles (héritage)
    insert(conn, "INSERT INTO conducteur(utilisateur_id) VALUES (?);", (conducteur_id,))
    insert(conn, "INSERT INTO passager(utilisateur_id) VALUES (?);", (passager_id,))

    # =========================================================
    # 2. Documents du conducteur
    # =========================================================

    # Permis (document + permis)
    permis_doc_id = insert(
        conn,
        """
        INSERT INTO document(conducteur_id, fichier_url, date_expiration, statut, type)
        VALUES (?, ?, ?, ?, 'PERMIS');
        """,
        (
            conducteur_id,
            "https://exemple.com/permis.pdf",
            "2034-06-01",
            "VALIDE",
        ),
    )

    insert(
        conn,
        """
        INSERT INTO permis(document_id, numero, date_obtention)
        VALUES (?, ?, ?);
        """,
        (permis_doc_id, "PERMIS-AAA-123", "2010-06-01"),
    )

    # Contrôle technique (document + controle_technique)
    ct_doc_id = insert(
        conn,
        """
        INSERT INTO document(conducteur_id, fichier_url, date_expiration, statut, type)
        VALUES (?, ?, ?, ?, 'CONTROLE_TECHNIQUE');
        """,
        (
            conducteur_id,
            "https://exemple.com/controle_technique.pdf",
            "2026-01-15",
            "VALIDE",
        ),
    )

    insert(
        conn,
        """
        INSERT INTO controle_technique(document_id, numero_immatriculation, date_controle)
        VALUES (?, ?, ?);
        """,
        (ct_doc_id, "AB-123-CD", "2024-12-01"),
    )

    # =========================================================
    # 3. Trajet proposé par le conducteur
    # =========================================================
    now = datetime.now()
    date_depart = (now + timedelta(days=2)).strftime("%Y-%m-%d %H:%M") # Formatage propre

    trajet_id = insert(
        conn,
        """
        INSERT INTO trajet(
            conducteur_id, depart, arrivee, date_heure,
            prix, nb_places_total, nb_places_disponibles,
            consignes, statut
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            conducteur_id,
            "Paris",
            "Lyon",
            date_depart,
            35.0,
            3,
            3,
            "Bagages cabine uniquement.",
            "PUBLIE",
        ),
    )

    # Étapes du trajet
    insert(
        conn,
        """
        INSERT INTO etape(trajet_id, ordre, lieu)
        VALUES (?, ?, ?);
        """,
        (trajet_id, 1, "Aire de Nemours"),
    )

    insert(
        conn,
        """
        INSERT INTO etape(trajet_id, ordre, lieu)
        VALUES (?, ?, ?);
        """,
        (trajet_id, 2, "Aire de Mâcon"),
    )

    # =========================================================
    # 4. Réservation du passager
    # =========================================================

    insert(
        conn,
        """
        INSERT INTO reservation(passager_id, trajet_id, nb_places, statut)
        VALUES (?, ?, ?, ?);
        """,
        (passager_id, trajet_id, 1, "ACTIVE"),
    )
    
    # Mise à jour du stock places (Important pour la cohérence !)
    conn.execute(
        "UPDATE trajet SET nb_places_disponibles = nb_places_disponibles - 1 WHERE id = ?", 
        (trajet_id,)
    )

    # =========================================================
    # 5. Avis du passager
    # =========================================================

    insert(
        conn,
        """
        INSERT INTO avis(
            passager_id, trajet_id, note,
            commentaire, anonyme, date_publication
        )
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        (
            passager_id,
            trajet_id,
            5,
            "Très bon trajet, conducteur ponctuel.",
            0,  # anonyme = False
            datetime.now().isoformat(timespec="seconds"),
        ),
    )

    conn.commit()
    conn.close()

    print("Données de test insérées avec succès !")