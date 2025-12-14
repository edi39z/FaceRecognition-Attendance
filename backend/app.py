from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import base64
import cv2
import numpy as np
import os
import re
import bcrypt
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from urllib.parse import urlparse

# --- LIBRARY BARU: INSIGHTFACE (MOBILEFACENET) ---
from insightface.app import FaceAnalysis
from numpy.linalg import norm

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# 1. INISIALISASI MODEL MOBILEFACENET
# ==========================================
print("⏳ Sedang memuat model MobileFaceNet (InsightFace)...")
# 'buffalo_s' adalah paket ringan berisi MobileFaceNet (Recognition) + SCRFD (Detection)
# ctx_id=-1 artinya menggunakan CPU (Ubah ke 0 jika pakai GPU NVIDIA)
app_face = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app_face.prepare(ctx_id=-1, det_size=(640, 640))
print("✅ Model MobileFaceNet berhasil dimuat!")

# ==========================================
# 2. FUNGSI MATEMATIKA (COSINE SIMILARITY)
# ==========================================
def compute_similarity(feat1, feat2):
    """
    Menghitung kemiripan antara dua wajah.
    MobileFaceNet menggunakan Cosine Similarity.
    Range output: -1.0 s/d 1.0 (Semakin mendekati 1.0, semakin mirip).
    """
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
    return sim

# ==========================================
# 3. KONEKSI DATABASE
# ==========================================
def get_db_connection():
    try:
        database_url = os.environ.get("DATABASE_URL")
        
        if database_url:
            # print(f"Connecting to Neon database...")
            conn = psycopg2.connect(database_url, connect_timeout=5)
        else:
            def strip_quotes(value):
                if value and len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                    return value[1:-1]
                if value and len(value) >= 2 and value[0] == "'" and value[-1] == "'":
                    return value[1:-1]
                return value

            DB_HOST = strip_quotes(os.environ.get("DB_HOST", "localhost"))
            DB_NAME = strip_quotes(os.environ.get("DB_NAME", "face_recognition"))
            DB_USER = strip_quotes(os.environ.get("DB_USER", "postgres"))
            DB_PASS = strip_quotes(os.environ.get("DB_PASS", "password"))
            
            # print(f"Connecting to localhost: {DB_HOST}:{DB_NAME} with user {DB_USER}")
            conn = psycopg2.connect(
                host=DB_HOST, 
                database=DB_NAME, 
                user=DB_USER, 
                password=DB_PASS,
                connect_timeout=5,
                client_encoding='UTF8'
            )
        
        return conn
    except psycopg2.OperationalError as e:
        print(f" PostgreSQL connection error: {e}")
        app.logger.error(f"Flask: PostgreSQL connection error: {e}")
        return None
    except Exception as e:
        print(f" Unexpected error connecting to database: {e}")
        app.logger.error(f"Flask: Unexpected error: {e}")
        return None

# ==========================================
# 4. ENDPOINT UTAMA
# ==========================================

# --- Endpoint Absensi (MobileFaceNet) ---
@app.route('/attendance', methods=['POST'])
def attendance():
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # 1. Decode Gambar Base64
        image_b64 = data['image'].split(',')[-1]
        image_data = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_data, np.uint8)
        image_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_cv2 is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # 2. Deteksi & Ekstraksi Fitur dengan MobileFaceNet
        faces = app_face.get(image_cv2)
        
        if len(faces) == 0:
            return jsonify({'error': 'Wajah tidak terdeteksi. Harap posisi wajah tegak lurus.'}), 404
        
        # Ambil wajah pertama (yang paling dominan/besar score-nya)
        current_face_emb = faces[0].embedding

        # 3. Ambil Data Karyawan dari Database
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'}), 503
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute("SELECT nip, nama, face_embedding FROM \"Karyawan\" WHERE face_embedding IS NOT NULL")
            all_karyawan = cursor.fetchall()

            if not all_karyawan:
                 return jsonify({'error': 'Database wajah kosong. Harap registrasi wajah dulu.'}), 404

            max_similarity = -1.0
            best_match_user = None

            # --- THRESHOLD (AMBANG BATAS) ---
            # Untuk MobileFaceNet, 0.5 adalah angka yang seimbang.
            # < 0.4 : Terlalu ketat (susah absen)
            # > 0.6 : Terlalu longgar (bisa salah orang)
            THRESHOLD = 0.5 

            # 4. Loop Pencocokan (Matching)
            for user in all_karyawan:
                try:
                    str_encoding = user['face_embedding']
                    # Bersihkan format string dan convert ke Numpy Array
                    # Asumsi format di DB: "[0.123, -0.456, ...]"
                    clean_str = str_encoding.strip().replace('\n', '')
                    db_emb_list = [float(x) for x in clean_str.strip('[]').split(',')]
                    db_emb_np = np.array(db_emb_list)

                    # Hitung Skor Kemiripan
                    sim = compute_similarity(current_face_emb, db_emb_np)

                    if sim > max_similarity:
                        max_similarity = sim
                        best_match_user = user

                except Exception as e:
                    # Skip jika ada data corrupt di DB
                    continue
            
            # 5. Keputusan Akhir
            if max_similarity > THRESHOLD and best_match_user:
                return jsonify({
                    'message': 'Face recognized',
                    'name': best_match_user['nama'],
                    'nip': best_match_user['nip'],
                    'similarity': float(max_similarity) # Opsional: untuk debug
                }), 200
            else:
                return jsonify({
                    'error': 'Wajah tidak dikenali', 
                    'similarity': float(max_similarity)
                }), 404

        except psycopg2.Error as db_err:
            app.logger.error(f"Flask DB Error: {str(db_err)}")
            return jsonify({'error': f'Database error: {str(db_err)}'}), 500
        finally:
            if conn:
                cursor.close()
                conn.close()

    except Exception as e:
        app.logger.error(f"Flask Error: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# --- Endpoint Pendaftaran Wajah (MobileFaceNet) ---
@app.route('/register-face', methods=['POST'])
def register_face():
    data = request.get_json()

    if not data or 'nip' not in data or 'fotoWajah' not in data:
        return jsonify({'error': 'Data tidak lengkap: nip dan fotoWajah wajib diisi'}), 400

    try:
        image_b64_data_url = data['fotoWajah']
        if ',' in image_b64_data_url:
            header, image_b64 = image_b64_data_url.split(',', 1)
        else:
            image_b64 = image_b64_data_url

        image_data = base64.b64decode(image_b64)
        np_arr = np.frombuffer(image_data, np.uint8)
        image_cv2 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_cv2 is None:
            return jsonify({'error': 'Gagal decode gambar'}), 400

        # --- DETEKSI DENGAN MOBILEFACENET ---
        faces = app_face.get(image_cv2)

        if len(faces) == 0:
            return jsonify({'error': 'Wajah tidak terdeteksi. Pastikan pencahayaan cukup.'}), 400
        
        if len(faces) > 1:
            return jsonify({'error': 'Terdeteksi lebih dari satu wajah. Gunakan foto selfie sendiri.'}), 400

        # Ambil embedding (512 dimensi)
        # Convert ke list Python biasa agar bisa disimpan sebagai JSON/String di DB
        face_encoding_list = faces[0].embedding.tolist()

        return jsonify({
            'face_encoding': face_encoding_list,
            'message': 'Wajah berhasil diproses dengan MobileFaceNet'
        }), 200

    except Exception as e:
        app.logger.error(f"Register Error: {str(e)}")
        return jsonify({'error': f'Flask error: {str(e)}'}), 500

# --- Endpoint Login (Tidak Berubah) ---
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    def strip_quotes(value):
        if value and len(value) >= 2 and value[0] == '"' and value[-1] == '"':
            return value[1:-1]
        if value and len(value) >= 2 and value[0] == "'" and value[-1] == "'":
            return value[1:-1]
        return value

    admin_email = strip_quotes(os.getenv('ADMIN_EMAIL', ''))
    admin_password = strip_quotes(os.getenv('ADMIN_PASSWORD', ''))

    # Check admin
    if email == admin_email and password == admin_password:
        return jsonify({"message": "Login berhasil", "role": "admin"}), 200

    # Check user
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Tidak bisa konek ke database"}), 503

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute('SELECT nama, password, email FROM public."Karyawan" WHERE email = %s', (email,))
        row = cursor.fetchone()

        if row:
            nama = row['nama']
            hashed_password = row['password']
            try:
                if hashed_password is None:
                    return jsonify({"message": "Email atau password salah"}), 401
                
                password_bytes = password.encode('utf-8')
                if isinstance(hashed_password, str):
                    hashed_password_bytes = hashed_password.encode('utf-8')
                else:
                    hashed_password_bytes = hashed_password
                
                if bcrypt.checkpw(password_bytes, hashed_password_bytes):
                    return jsonify({"message": "Login berhasil", "role": "user", "nama": nama}), 200
                else:
                    return jsonify({"message": "Email atau password salah"}), 401
            except Exception as e:
                return jsonify({"message": "Email atau password salah"}), 401
        else:
            return jsonify({"message": "Email atau password salah"}), 401

    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500
    finally:
        cursor.close()
        conn.close()

# --- Health check ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'message': 'Flask server with MobileFaceNet is running'}), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Endpoints: /attendance, /register-face, /api/login")
    app.run(host='0.0.0.0', port=5000, debug=True)