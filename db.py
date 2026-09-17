"""
db.py
-----
MySQL data layer for PNEUMO-AI.

Handles:
- Database/table creation
- Patient (user) register/login
- Doctor register/login
- Saving AI diagnosis results
- Booking hospital tokens
- Doctor dashboard queries

Connection settings are read from .streamlit/secrets.toml:

    [mysql]
    host = "localhost"
    user = "root"
    password = "your-mysql-password"
    database = "pneumo_ai"
    port = 3306

If secrets.toml isn't set up, it falls back to environment variables
MYSQL_HOST / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE / MYSQL_PORT.
"""

import os
import hmac
import hashlib

import mysql.connector
import streamlit as st


# =========================================================
# CONNECTION CONFIG
# =========================================================

def _get_config():
    try:
        cfg = st.secrets["mysql"]
        return {
            "host": cfg.get("host", "localhost"),
            "user": cfg.get("user", "root"),
            "password": cfg.get("password", "23182318@Nishant"),
            "database": cfg.get("database", "pneumo_ai"),
            "port": int(cfg.get("port", 3306)),
        }
    except Exception:
        return {
            "host": os.environ.get("MYSQL_HOST", "localhost"),
            "user": os.environ.get("MYSQL_USER", "root"),
            "password": os.environ.get("MYSQL_PASSWORD", ""),
            "database": os.environ.get("MYSQL_DATABASE", "pneumo_ai"),
            "port": int(os.environ.get("MYSQL_PORT", 3306)),
        }


def get_connection():
    cfg = _get_config()
    return mysql.connector.connect(
        host=cfg["host"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
        port=cfg["port"],
    )


# =========================================================
# SCHEMA SETUP
# =========================================================

def init_db():
    """Creates the database (if missing) and all required tables."""
    cfg = _get_config()

    root_conn = mysql.connector.connect(
        host=cfg["host"], user=cfg["user"], password=cfg["password"], port=cfg["port"]
    )
    root_cursor = root_conn.cursor()
    root_cursor.execute(
        f"CREATE DATABASE IF NOT EXISTS `{cfg['database']}` "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )
    root_conn.commit()
    root_cursor.close()
    root_conn.close()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(150) NOT NULL,
            email VARCHAR(150) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            salt VARCHAR(64) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id INT AUTO_INCREMENT PRIMARY KEY,
            full_name VARCHAR(150) NOT NULL,
            email VARCHAR(150) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            salt VARCHAR(64) NOT NULL,
            specialization VARCHAR(150) DEFAULT 'Pulmonologist',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnoses (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            result VARCHAR(20) NOT NULL,
            confidence FLOAT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            id INT AUTO_INCREMENT PRIMARY KEY,
            diagnosis_id INT NOT NULL,
            user_id INT NOT NULL,
            hospital_name VARCHAR(255) NOT NULL,
            hospital_address VARCHAR(255),
            hospital_place_id VARCHAR(255),
            token_number VARCHAR(20) NOT NULL,
            status VARCHAR(20) DEFAULT 'BOOKED',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id) ON DELETE CASCADE,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS symptom_checks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            symptoms TEXT NOT NULL,
            predicted_disease VARCHAR(255) NOT NULL,
            severity VARCHAR(100) NOT NULL,
            confidence FLOAT NOT NULL,
            recommended_doctor VARCHAR(255) NOT NULL,
            doctor_advice TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()


# =========================================================
# PASSWORD HASHING (PBKDF2 - stdlib only, no extra dependency)
# =========================================================

def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


def _make_salt() -> str:
    return os.urandom(16).hex()


# =========================================================
# PATIENT (USER) AUTH
# =========================================================

def register_user(full_name, email, password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return False, "An account with this email already exists."

    salt = _make_salt()
    password_hash = _hash_password(password, salt)

    cursor.execute(
        "INSERT INTO users (full_name, email, password_hash, salt) VALUES (%s, %s, %s, %s)",
        (full_name, email, password_hash, salt),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return True, "Account created successfully. Please log in."


def login_user(email, password):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return None, "No account found with this email."

    expected_hash = _hash_password(password, row["salt"])
    if not hmac.compare_digest(expected_hash, row["password_hash"]):
        return None, "Incorrect password."

    return row, "Login successful."


# =========================================================
# DOCTOR AUTH
# =========================================================

def register_doctor(full_name, email, password, specialization):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM doctors WHERE email = %s", (email,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return False, "A doctor account with this email already exists."

    salt = _make_salt()
    password_hash = _hash_password(password, salt)

    cursor.execute(
        """INSERT INTO doctors (full_name, email, password_hash, salt, specialization)
           VALUES (%s, %s, %s, %s, %s)""",
        (full_name, email, password_hash, salt, specialization),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return True, "Doctor account created successfully. Please log in."


def login_doctor(email, password):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM doctors WHERE email = %s", (email,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        return None, "No doctor account found with this email."

    expected_hash = _hash_password(password, row["salt"])
    if not hmac.compare_digest(expected_hash, row["password_hash"]):
        return None, "Incorrect password."

    return row, "Login successful."


# =========================================================
# DIAGNOSES
# =========================================================

def save_diagnosis(user_id, result, confidence):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO diagnoses (user_id, result, confidence) VALUES (%s, %s, %s)",
        (user_id, result, confidence),
    )
    conn.commit()
    diagnosis_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return diagnosis_id


# =========================================================
# TOKENS / APPOINTMENTS
# =========================================================

def book_token(diagnosis_id, user_id, hospital_name, hospital_address, hospital_place_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """SELECT COUNT(*) FROM tokens
           WHERE hospital_place_id = %s AND DATE(created_at) = CURDATE()""",
        (hospital_place_id,),
    )
    count_today = cursor.fetchone()[0]
    token_number = f"T{count_today + 1:04d}"

    cursor.execute(
        """INSERT INTO tokens
               (diagnosis_id, user_id, hospital_name, hospital_address, hospital_place_id, token_number)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (diagnosis_id, user_id, hospital_name, hospital_address, hospital_place_id, token_number),
    )
    conn.commit()
    cursor.close()
    conn.close()
    return token_number


def get_user_tokens(user_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        """SELECT t.*, d.result, d.confidence
           FROM tokens t
           JOIN diagnoses d ON t.diagnosis_id = d.id
           WHERE t.user_id = %s
           ORDER BY t.created_at DESC""",
        (user_id,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


# =========================================================
# DOCTOR DASHBOARD DATA
# =========================================================

def get_pneumonia_patients_with_tokens():
    """
    Every patient whose diagnosis was PNEUMONIA, joined with any
    token they've booked (LEFT JOIN so patients with no token yet
    still show up).
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            u.id AS user_id,
            u.full_name,
            u.email,
            d.id AS diagnosis_id,
            d.result,
            d.confidence,
            d.created_at AS diagnosed_at,
            t.token_number,
            t.hospital_name,
            t.hospital_address,
            t.status,
            t.created_at AS token_created_at
        FROM diagnoses d
        JOIN users u ON d.user_id = u.id
        LEFT JOIN tokens t ON t.diagnosis_id = d.id
        WHERE d.result = 'PNEUMONIA'
        ORDER BY d.created_at DESC
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


# =========================================================
# SYMPTOM CHECKS & DOCTOR RECOMMENDATIONS
# =========================================================

def save_symptom_check(user_id, symptoms, predicted_disease, severity, confidence, recommended_doctor, doctor_advice=""):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """INSERT INTO symptom_checks
           (user_id, symptoms, predicted_disease, severity, confidence, recommended_doctor, doctor_advice)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (user_id, symptoms, predicted_disease, severity, confidence, recommended_doctor, doctor_advice),
    )
    conn.commit()
    check_id = cursor.lastrowid
    cursor.close()
    conn.close()
    return check_id


def get_user_symptom_checks(user_id):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute(
        """SELECT * FROM symptom_checks
           WHERE user_id = %s
           ORDER BY created_at DESC""",
        (user_id,),
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def get_all_symptom_checks():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            sc.*,
            u.full_name,
            u.email
        FROM symptom_checks sc
        JOIN users u ON sc.user_id = u.id
        ORDER BY sc.created_at DESC
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()