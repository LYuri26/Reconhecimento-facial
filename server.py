from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from config import Config
from app.database import Database
from app.face_recognition import FaceRecognition
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]
Config.init_app(app)

db = Database()
fr = FaceRecognition()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "jpg",
        "jpeg",
        "png",
    }


def create_person_folder(nome, sobrenome, pessoa_id):
    """Cria pasta para armazenar as imagens da pessoa"""
    nome_pasta = f"{nome}_{sobrenome}_{pessoa_id}".replace(" ", "_").lower()
    pessoa_folder = os.path.join(app.config["UPLOAD_FOLDER"], nome_pasta)
    os.makedirs(pessoa_folder, exist_ok=True)
    return nome_pasta, pessoa_folder


def generate_image_filename(nome, sobrenome):
    """Gera nome único para a imagem"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return secure_filename(f"{nome}_{sobrenome}_{timestamp}.jpg")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        nome = request.form.get("nome", "").strip()
        sobrenome = request.form.get("sobrenome", "").strip()
        observacoes = request.form.get("observacoes", "").strip()
        imagens = request.files.getlist("imagens")

        if not nome or not sobrenome:
            flash("Nome e sobrenome são obrigatórios!", "danger")
            return redirect(url_for("index"))

        if len(imagens) < 1:
            flash("Selecione pelo menos uma imagem!", "danger")
            return redirect(url_for("index"))

        try:
            # Insere pessoa no banco de dados
            pessoa_id = db.inserir_pessoa(nome, sobrenome, observacoes)

            # Cria pasta para a pessoa
            nome_pasta, pessoa_folder = create_person_folder(nome, sobrenome, pessoa_id)

            # Processa cada imagem
            for img in imagens:
                if img and allowed_file(img.filename):
                    filename = generate_image_filename(nome, sobrenome)
                    save_path = os.path.join(pessoa_folder, filename)
                    img.save(save_path)

                    # Extrai encodings faciais
                    encodings = fr.extract_face_encodings(save_path)
                    if encodings:
                        relative_path = os.path.join(nome_pasta, filename)
                        db.inserir_imagem(pessoa_id, relative_path, str(encodings[0]))
                    else:
                        os.remove(save_path)
                        flash(
                            "Não foi possível detectar um rosto em uma das imagens!",
                            "warning",
                        )

            flash("Cadastro realizado com sucesso!", "success")
            return redirect(url_for("cadastro_concluido", id=pessoa_id))

        except Exception as e:
            flash(f"Erro ao processar cadastro: {str(e)}", "danger")
            return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/cadastro-concluido/<int:id>")
def cadastro_concluido(id):
    pessoa = db.obter_pessoa(id)
    if not pessoa:
        flash("Pessoa não encontrada!", "danger")
        return redirect(url_for("index"))
    return render_template("cadastro-concluido.html", pessoa=pessoa)


@app.route("/api/reconhecer", methods=["POST"])
def api_reconhecer():
    if "imagem" not in request.files:
        return jsonify({"erro": "Nenhuma imagem enviada"}), 400

    file = request.files["imagem"]
    if file.filename == "":
        return jsonify({"erro": "Nenhuma imagem selecionada"}), 400

    if file and allowed_file(file.filename):
        temp_path = os.path.join(app.config["TEMP_FOLDER"], "temp_recognition.jpg")
        file.save(temp_path)

        try:
            img = cv2.imread(temp_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = fr.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            if len(faces) == 0:
                return jsonify({"sucesso": False, "mensagem": "Nenhum rosto detectado"})

            (x, y, w, h) = faces[0]
            face = gray[y : y + h, x : x + w]
            face = cv2.resize(face, (220, 220))

            # Reconhecimento
            pessoa_id, confianca = fr.reconhecer_face(temp_path, db)

            if pessoa_id:
                pessoa = db.obter_pessoa(pessoa_id)
                return jsonify(
                    {
                        "sucesso": True,
                        "pessoa": {
                            "id": pessoa_id,
                            "nome": pessoa["nome"],
                            "sobrenome": pessoa["sobrenome"],
                        },
                        "face": {
                            "x": int(x),
                            "y": int(y),
                            "width": int(w),
                            "height": int(h),
                        },
                        "confianca": float(confianca),
                        "data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            else:
                return jsonify(
                    {
                        "sucesso": False,
                        "mensagem": "Pessoa não reconhecida",
                        "face": {
                            "x": int(x),
                            "y": int(y),
                            "width": int(w),
                            "height": int(h),
                        },
                    }
                )

        except Exception as e:
            return jsonify({"erro": str(e)}), 500

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return jsonify({"erro": "Formato de arquivo não permitido"}), 400


@app.route("/dashboard")
def dashboard():
    pessoas = db.listar_pessoas()
    total_reconhecimentos = db.contar_reconhecimentos()
    return render_template(
        "dashboard.html", pessoas=pessoas, total_reconhecimentos=total_reconhecimentos
    )


@app.route("/logs")
def logs():
    registros = db.listar_logs()
    return render_template("logs.html", logs=registros)


@app.route("/pessoa/<int:id>")
def ver_pessoa(id):
    pessoa = db.obter_pessoa(id)
    if not pessoa:
        flash("Pessoa não encontrada!", "danger")
        return redirect(url_for("dashboard"))

    imagens = db.listar_imagens(id)
    reconhecimentos = db.listar_reconhecimentos(id)

    return render_template(
        "ver-pessoa.html",
        pessoa=pessoa,
        imagens=imagens,
        reconhecimentos=reconhecimentos,
    )


@app.route("/pessoa/deletar/<int:id>", methods=["POST"])
def deletar_pessoa(id):
    try:
        pessoa = db.obter_pessoa(id)
        if not pessoa:
            flash("Pessoa não encontrada!", "danger")
            return redirect(url_for("dashboard"))

        # Remove pasta da pessoa
        nome_pasta = f"{pessoa['nome']}_{pessoa['sobrenome']}_{id}".replace(
            " ", "_"
        ).lower()
        pessoa_folder = os.path.join(app.config["UPLOAD_FOLDER"], nome_pasta)

        if os.path.exists(pessoa_folder):
            for filename in os.listdir(pessoa_folder):
                file_path = os.path.join(pessoa_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Erro ao deletar {file_path}: {e}")

            os.rmdir(pessoa_folder)

        # Remove do banco de dados
        db.deletar_pessoa(id)
        flash("Pessoa removida com sucesso!", "success")
    except Exception as e:
        flash(f"Erro ao remover pessoa: {str(e)}", "danger")

    return redirect(url_for("dashboard"))


@app.route("/reconhecer")
def reconhecer():
    return render_template("reconhecer.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
