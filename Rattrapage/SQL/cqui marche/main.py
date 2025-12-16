# main.py
import sqlite3

DB_PATH = "covoiturage.db"


def doquery(conn, sql):
    conn.execute(sql)
    conn.commit()


def check(conn):
    conn.execute("SELECT 1;")
    print("Bien connecté à la BD SQLite !")


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    check(conn)

    # Suppression des tables (Ordre important pour les FK)
    tables = [
        "controle_technique",
        "permis",
        "document",
        "avis",
        "reservation",
        "etape",
        "trajet",
        "passager",
        "conducteur",
        "utilisateur",
    ]
    for t in tables:
        doquery(conn, f"DROP TABLE IF EXISTS {t};")

    # ================
    # UTILISATEUR
    # ================
    doquery(
        conn,
        """
        CREATE TABLE utilisateur (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            mot_de_passe_hash TEXT NOT NULL,
            prenom TEXT NOT NULL,
            nom TEXT NOT NULL,
            date_naissance TEXT NOT NULL
        );
        """,
    )

    # PASSAGER
    doquery(
        conn,
        """
        CREATE TABLE passager (
            utilisateur_id INTEGER PRIMARY KEY,
            FOREIGN KEY (utilisateur_id) REFERENCES utilisateur(id) ON DELETE CASCADE
        );
        """,
    )

    # CONDUCTEUR
    doquery(
        conn,
        """
        CREATE TABLE conducteur (
            utilisateur_id INTEGER PRIMARY KEY,
            FOREIGN KEY (utilisateur_id) REFERENCES utilisateur(id) ON DELETE CASCADE
        );
        """,
    )

    # ================
    # TRAJET
    # ================
    doquery(
        conn,
        """
        CREATE TABLE trajet (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conducteur_id INTEGER NOT NULL,
            depart TEXT NOT NULL,
            arrivee TEXT NOT NULL,
            date_heure TEXT NOT NULL,
            prix REAL NOT NULL CHECK (prix >= 0),
            nb_places_total INTEGER NOT NULL,
            nb_places_disponibles INTEGER NOT NULL,
            consignes TEXT,
            statut TEXT NOT NULL DEFAULT 'PUBLIE'
                CHECK (statut IN ('PUBLIE','ANNULE','EFFECTUE','CLOTURE')),
            CHECK (nb_places_total > 0),
            CHECK (nb_places_disponibles >= 0),
            CHECK (nb_places_disponibles <= nb_places_total),
            FOREIGN KEY (conducteur_id) REFERENCES conducteur(utilisateur_id) ON DELETE CASCADE
        );
        """,
    )
    
    # Trigger : Verrouillage trajet clos
    doquery(
        conn,
        """
        CREATE TRIGGER IF NOT EXISTS trg_verrouillage_trajet_clos
        BEFORE UPDATE ON trajet
        WHEN OLD.statut IN ('EFFECTUE', 'ANNULE', 'CLOTURE')
             AND NEW.statut = OLD.statut
        BEGIN
            SELECT RAISE(ABORT, 'INTERDIT: Impossible de modifier les détails d''un trajet clos.');
        END;
        """
    )

    # ================
    # ETAPE
    # ================
    doquery(
        conn,
        """
        CREATE TABLE etape (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trajet_id INTEGER NOT NULL,
            ordre INTEGER NOT NULL,
            lieu TEXT NOT NULL,
            FOREIGN KEY (trajet_id) REFERENCES trajet(id) ON DELETE CASCADE,
            UNIQUE (trajet_id, ordre)
        );
        """,
    )

    # ================
    # RESERVATION
    # ================
    doquery(
        conn,
        """
        CREATE TABLE reservation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            passager_id INTEGER NOT NULL,
            trajet_id INTEGER NOT NULL,
            nb_places INTEGER NOT NULL CHECK (nb_places > 0),
            statut TEXT NOT NULL DEFAULT 'ACTIVE'
                CHECK (statut IN ('ACTIVE','ANNULEE_PASSAGER','ANNULEE_CONDUCTEUR','EFFECTUEE')),
            UNIQUE (passager_id, trajet_id),
            FOREIGN KEY (passager_id) REFERENCES passager(utilisateur_id) ON DELETE CASCADE,
            FOREIGN KEY (trajet_id) REFERENCES trajet(id) ON DELETE CASCADE
        );
        """,
    )
    
    # Trigger : Anti auto-réservation
    doquery(
        conn,
        """
        CREATE TRIGGER IF NOT EXISTS trg_anti_auto_reservation
        BEFORE INSERT ON reservation
        BEGIN
            SELECT CASE
                WHEN EXISTS (
                    SELECT 1 FROM trajet
                    WHERE id = NEW.trajet_id
                    AND conducteur_id = NEW.passager_id
                )
                THEN RAISE(ABORT, 'INTERDIT: Le conducteur ne peut pas être passager de son propre trajet.')
            END;
        END;
        """
    )

    # ================
    # AVIS
    # ================
    doquery(
        conn,
        """
        CREATE TABLE avis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            passager_id INTEGER NOT NULL,
            trajet_id INTEGER NOT NULL,
            note INTEGER NOT NULL CHECK (note BETWEEN 1 AND 5),
            commentaire TEXT,
            anonyme INTEGER NOT NULL CHECK (anonyme IN (0,1)),
            date_publication TEXT NOT NULL,
            UNIQUE (passager_id, trajet_id),
            FOREIGN KEY (passager_id) REFERENCES passager(utilisateur_id) ON DELETE CASCADE,
            FOREIGN KEY (trajet_id) REFERENCES trajet(id) ON DELETE CASCADE
        );
        """,
    )

    # ================
    # DOCUMENT
    # ================
    doquery(
        conn,
        """
        CREATE TABLE document (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conducteur_id INTEGER NOT NULL,
            fichier_url TEXT NOT NULL,
            date_expiration TEXT,
            statut TEXT NOT NULL DEFAULT 'EN_COURS' CHECK (statut IN ('EN_COURS','VALIDE','REFUSE')),
            type TEXT NOT NULL CHECK (type IN ('PERMIS','CONTROLE_TECHNIQUE')),
            FOREIGN KEY (conducteur_id) REFERENCES conducteur(utilisateur_id) ON DELETE CASCADE
        );
        """,
    )

    # PERMIS
    doquery(
        conn,
        """
        CREATE TABLE permis (
            document_id INTEGER PRIMARY KEY,
            numero TEXT NOT NULL,
            date_obtention TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES document(id) ON DELETE CASCADE
        );
        """,
    )

    # CONTROLE TECHNIQUE
    doquery(
        conn,
        """
        CREATE TABLE controle_technique (
            document_id INTEGER PRIMARY KEY,
            numero_immatriculation TEXT NOT NULL,
            date_controle TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES document(id) ON DELETE CASCADE
        );
        """,
    )

    # Indexes
    doquery(conn, "CREATE INDEX idx_trajet_conducteur_id ON trajet(conducteur_id);")
    doquery(conn, "CREATE INDEX idx_reservation_passager_id ON reservation(passager_id);")
    doquery(conn, "CREATE INDEX idx_reservation_trajet_id ON reservation(trajet_id);")
    doquery(conn, "CREATE INDEX idx_avis_passager_id ON avis(passager_id);")
    doquery(conn, "CREATE INDEX idx_avis_trajet_id ON avis(trajet_id);")
    doquery(conn, "CREATE INDEX idx_document_conducteur_id ON document(conducteur_id);")

    print("Toutes les tables ont été créées !")

    conn.close()