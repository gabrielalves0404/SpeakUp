import asyncio
import difflib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import deque

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
import speech_recognition as sr
from googletrans import Translator

# ============================================================
# ⚙️ CONFIGURAÇÕES
# ============================================================
sample_rate = 16000            # 16 kHz é o ideal para reconhecimento de fala
arquivo_audio = "output.wav"   # arquivo onde a gravação é salva
arquivo_ranking = "ranking.json"
limiar = 0.75                  # similaridade mínima (0 a 1) para aceitar
config = {"limite_voz": 300}    # volume mínimo para considerar que há voz (calibrado no início)
tentativas_por_palavra = 2

# Idioma do jogo: inglês
codigo_idioma = "en"          # código do googletrans
codigo_fala = "en-US"         # código do reconhecimento de fala
nome_idioma = "Inglês 🇺🇸"

# Níveis de dificuldade
niveis = {
    "1": {
        "nome": "Fácil 🟢",
        "vidas": 5,
        "segundos": 5,
        "pontos": 10,
        "palavras": ["gato", "cachorro", "casa", "água", "livro", "sol",
                     "lua", "pão", "leite", "amigo", "maçã", "escola"],
    },
    "2": {
        "nome": "Médio 🟡",
        "vidas": 3,
        "segundos": 4,
        "pontos": 20,
        "palavras": ["janela", "cadeira", "borboleta", "travesseiro", "chave",
                     "cozinha", "sorriso", "montanha", "floresta",
                     "aniversário", "cachoeira", "nuvem"],
    },
    "3": {
        "nome": "Difícil 🔴",
        "vidas": 2,
        "segundos": 3,
        "pontos": 30,
        "palavras": ["desenvolvimento", "ambiente", "conhecimento",
                     "sobrancelha", "inesperado", "vizinhança", "desafio",
                     "orgulhoso", "pesadelo", "comportamento", "ferramenta",
                     "pensamento"],
    },
}

tradutor = Translator()
cache_traducoes = {}  # evita traduzir a mesma palavra duas vezes

# Traduções de reserva: usadas se o googletrans falhar ou devolver algo errado
reserva = {
    "gato": "cat", "cachorro": "dog", "casa": "house", "água": "water",
    "livro": "book", "sol": "sun", "lua": "moon", "pão": "bread",
    "leite": "milk", "amigo": "friend", "maçã": "apple", "escola": "school",
    "janela": "window", "cadeira": "chair", "borboleta": "butterfly",
    "travesseiro": "pillow", "chave": "key", "cozinha": "kitchen",
    "sorriso": "smile", "montanha": "mountain", "floresta": "forest",
    "aniversário": "birthday", "cachoeira": "waterfall", "nuvem": "cloud",
    "desenvolvimento": "development", "ambiente": "environment",
    "conhecimento": "knowledge", "sobrancelha": "eyebrow",
    "inesperado": "unexpected", "vizinhança": "neighborhood",
    "desafio": "challenge", "orgulhoso": "proud", "pesadelo": "nightmare",
    "comportamento": "behavior", "ferramenta": "tool", "pensamento": "thought",
}


# ============================================================
# 🎨 INTERFACE
# ============================================================
def mostrar_titulo():
    print(r"""
  ╔══════════════════════════════════════════════╗
  ║   🎤  FALE A TRADUÇÃO  -  O JOGO  🌍         ║
  ╠══════════════════════════════════════════════╣
  ║  Veja a palavra em português e FALE a        ║
  ║  tradução em inglês!                          ║
  ║  Cuidado: errar custa uma vida ❤️            ║
  ╚══════════════════════════════════════════════╝
    """)


def mostrar_status(vidas, pontos, sequencia):
    print(f"\n  {'❤️ ' * vidas}  ⭐ Pontos: {pontos}   🔥 Sequência: {sequencia}")


def mostrar_instrucoes():
    print("  📖 COMO JOGAR")
    print("   • ENTER → gravar sua voz")
    print("   • d     → pedir uma dica (vale metade dos pontos)")
    print("   • p     → pular a palavra (perde a sequência)")
    print("   • s     → sair do jogo")
    print("   • Cada acerto seguido aumenta seu bônus 🔥\n")


# ============================================================
# 🔤 TEXTO: normalização e comparação
# ============================================================
def normalizar(texto):
    """Deixa minúsculo, sem acentos e sem pontuação."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^\w\s]", " ", texto)   # \w aceita letras de qualquer alfabeto
    return re.sub(r"\s+", " ", texto).strip()


def similaridade(a, b):
    return difflib.SequenceMatcher(None, normalizar(a), normalizar(b)).ratio()


def melhor_similaridade(hipoteses, resposta):
    """Compara cada hipótese (e cada palavra dela) com a resposta correta."""
    melhor = 0
    for frase in hipoteses:
        for trecho in [frase] + frase.split():
            melhor = max(melhor, similaridade(trecho, resposta))
    return melhor


# ============================================================
# 🌍 TRADUÇÃO
# ============================================================
def traduzir_com_googletrans(palavra, codigo_idioma):
    """Traduz usando o googletrans (funciona nas versões síncrona e assíncrona)."""
    tradutor = Translator()
    if asyncio.iscoroutinefunction(tradutor.translate):
        # Versão assíncrona: abre um tradutor novo a cada chamada para
        # evitar o erro "Event loop is closed"
        async def pedir():
            async with Translator() as t:
                resultado = await t.translate(palavra, src="pt", dest=codigo_idioma)
            return resultado.text
        return asyncio.run(pedir())
    return tradutor.translate(palavra, src="pt", dest=codigo_idioma).text


def traduzir(palavra, codigo_idioma):
    chave = (palavra, codigo_idioma)
    if chave in cache_traducoes:
        return cache_traducoes[chave]

    try:
        texto = traduzir_com_googletrans(palavra, codigo_idioma)
        # Se voltou igual à palavra original, a tradução falhou
        if normalizar(texto) == normalizar(palavra):
            raise ValueError("a tradução voltou igual à palavra original")
    except Exception:
        if palavra not in reserva:
            raise
        texto = reserva[palavra]
        if not config.get("avisou_reserva"):
            print("  ℹ️  O googletrans falhou; usando traduções de reserva.")
            config["avisou_reserva"] = True

    cache_traducoes[chave] = texto
    return texto


# ============================================================
# 🎙️ ÁUDIO E RECONHECIMENTO
# ============================================================
def volume_rms(trecho):
    """Mede o volume (energia média) de um trecho de áudio."""
    return float(np.sqrt(np.mean(trecho.astype(np.float32) ** 2)))


def calibrar_microfone():
    """Mede o ruído do ambiente para saber o que é voz e o que é silêncio."""
    print("  🔧 Calibrando o microfone... fique em silêncio por 1 segundo.")
    ruido = sd.rec(sample_rate, samplerate=sample_rate,
                   channels=1, dtype="int16")
    sd.wait()
    config["limite_voz"] = max(volume_rms(ruido) * 3, 150)
    print("  ✅ Microfone calibrado!\n")


def gravar(segundos_fala, espera=6):
    """
    Espera você começar a falar, grava e para sozinho quando você termina
    (ou quando passa de segundos_fala). Devolve True se ouviu voz.
    """
    print("  🎙️  Estou ouvindo... fale quando quiser!")
    tamanho_bloco = int(0.1 * sample_rate)   # blocos de 0,1 segundo
    antes = deque(maxlen=3)                  # guarda 0,3s antes da voz começar
    gravado = []
    falando = False
    blocos_silencio = 0
    inicio = time.time()

    with sd.InputStream(samplerate=sample_rate, channels=1,
                        dtype="int16", blocksize=tamanho_bloco) as entrada:
        while True:
            bloco, _ = entrada.read(tamanho_bloco)
            bloco = bloco[:, 0]
            tem_som = volume_rms(bloco) > config["limite_voz"]

            if not falando:
                antes.append(bloco)
                if tem_som:
                    falando = True
                    gravado = list(antes)
                    print("  🔴 Gravando...")
                elif time.time() - inicio > espera:
                    return False      # ninguém falou
            else:
                gravado.append(bloco)
                blocos_silencio = 0 if tem_som else blocos_silencio + 1
                # para após 0,7s de silêncio ou ao atingir o tempo máximo
                if blocos_silencio >= 7 or len(gravado) >= segundos_fala * 10:
                    break

    audio = np.concatenate(gravado)

    # Normaliza o volume: deixa a voz sempre em um nível bom para o Google
    pico = np.max(np.abs(audio))
    if pico > 0:
        audio = (audio.astype(np.float32) * (0.9 * 32767 / pico)).astype(np.int16)

    wav.write(arquivo_audio, sample_rate, audio)
    print("  ✅ Gravação concluída, estou reconhecendo...")
    return True


def reconhecer(codigo_fala):
    """Devolve uma lista de hipóteses do que foi dito (ou lista vazia)."""
    reconhecedor = sr.Recognizer()
    with sr.AudioFile(arquivo_audio) as fonte:
        audio = reconhecedor.record(fonte)

    try:
        resultado = reconhecedor.recognize_google(
            audio, language=codigo_fala, show_all=True)
    except sr.UnknownValueError:
        # O Google não entendeu nada (algumas versões lançam este erro)
        return []

    # Dependendo da versão da biblioteca, o resultado vem em formatos diferentes
    if isinstance(resultado, str):
        return [resultado]
    if isinstance(resultado, dict) and "alternative" in resultado:
        return [alt["transcript"] for alt in resultado["alternative"]
                if "transcript" in alt]
    return []


# ============================================================
# 🏆 PONTUAÇÃO E RANKING
# ============================================================
def classificar(acertos, total):
    """Dá uma nota em estrelas de acordo com a porcentagem de acertos."""
    if total == 0:
        return "—", "Nenhuma rodada jogada"
    porcentagem = acertos / total * 100
    if porcentagem >= 90:
        return "⭐⭐⭐⭐⭐", "Poliglota! 🧠"
    if porcentagem >= 70:
        return "⭐⭐⭐⭐", "Muito bom! 👏"
    if porcentagem >= 50:
        return "⭐⭐⭐", "Bom, continue praticando 💪"
    if porcentagem >= 30:
        return "⭐⭐", "Dá pra melhorar 🙂"
    return "⭐", "Treine mais! 📚"


def carregar_ranking():
    if os.path.exists(arquivo_ranking):
        try:
            with open(arquivo_ranking, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def salvar_no_ranking(nome, pontos, nivel):
    ranking = carregar_ranking()
    ranking.append({"nome": nome, "pontos": pontos, "nivel": nivel})
    ranking.sort(key=lambda item: item["pontos"], reverse=True)
    ranking = ranking[:5]  # guarda só o top 5
    with open(arquivo_ranking, "w", encoding="utf-8") as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)
    return ranking


def mostrar_ranking(ranking):
    print("\n  🏅 TOP 5 JOGADORES")
    print("  " + "─" * 44)
    posicoes = ["🥇", "🥈", "🥉", "4º", "5º"]
    for i, item in enumerate(ranking):
        print(f"  {posicoes[i]} {item['nome']:<12} {item['pontos']:>4} pts"
              f"  {item['nivel']}")


# ============================================================
# 🎮 RODADA
# ============================================================
def dica(resposta):
    """Mostra a primeira letra e o tamanho da palavra: C _ _"""
    letras = ["_"] * len(resposta)
    letras[0] = resposta[0].upper()
    return " ".join(letras)


def jogar_rodada(palavra, resposta, nome_idioma, codigo_fala, segundos):
    """
    Retorna uma tupla: (resultado, usou_dica)
    resultado pode ser 'acertou', 'errou', 'pulou' ou 'sair'
    """
    print("\n  ┌" + "─" * 44 + "┐")
    print(f"  │  Como se diz  ➜  {palavra.upper():<22}      │")
    print(f"  │  em {nome_idioma:<38} │")
    print("  └" + "─" * 44 + "┘")

    usou_dica = False
    tentativas = 0

    while tentativas < tentativas_por_palavra:
        comando = input("  [ENTER = falar | d = dica | p = pular | s = sair] ")
        comando = comando.strip().lower()

        if comando == "s":
            return "sair", usou_dica
        if comando == "p":
            print(f"  ⏭️  Resposta: {resposta}")
            return "pulou", usou_dica
        if comando == "d":
            usou_dica = True
            print(f"  💡 Dica: {dica(resposta)}")
            continue

        if not gravar(segundos):
            print("  🔇 Não ouvi nada. Verifique o microfone e tente de novo.")
            continue

        try:
            hipoteses = reconhecer(codigo_fala)
        except sr.RequestError as erro:
            print(f"  ⚠️  Erro no serviço de voz (internet?): {erro}")
            continue

        if not hipoteses:
            print("  🤷 A fala não pôde ser reconhecida. Tente de novo.")
            continue

        print(f"  👂 Você disse: \"{hipoteses[0]}\"")
        score = melhor_similaridade(hipoteses, resposta)
        print(f"  📊 Similaridade: {score * 100:.0f}%")

        if score >= limiar:
            print("  🎉 Correto!")
            return "acertou", usou_dica

        tentativas += 1
        if tentativas < tentativas_por_palavra:
            print("  ❌ Quase! Tente mais uma vez.")

    print(f"  ❌ Errou! A resposta era: {resposta}")
    return "errou", usou_dica


# ============================================================
# 🚀 PROGRAMA PRINCIPAL
# ============================================================
def escolher_nivel():
    print("\n  🎚️  Escolha a dificuldade:")
    for numero, dados in niveis.items():
        print(f"   {numero}) {dados['nome']}  "
              f"({dados['vidas']} vidas, {dados['segundos']}s no máximo para falar, "
              f"{dados['pontos']} pts por palavra)")
    while True:
        escolha = input("  Nível (1-3): ").strip()
        if escolha in niveis:
            return niveis[escolha]
        print("  Opção inválida.")


def jogar():
    mostrar_titulo()
    nome = input("  👤 Qual é o seu nome? ").strip() or "Jogador"
    print(f"\n  Olá, {nome}! 👋\n")

    nivel = escolher_nivel()
    print()
    mostrar_instrucoes()
    calibrar_microfone()

    vidas = nivel["vidas"]
    pontos = 0
    sequencia = 0
    acertos = 0
    rodadas = 0

    palavras = nivel["palavras"][:]
    random.shuffle(palavras)

    for palavra in palavras:
        if vidas <= 0:
            break

        try:
            resposta = traduzir(palavra, codigo_idioma)
        except Exception as erro:
            print(f"  ⚠️  Não consegui traduzir '{palavra}': {erro}")
            continue

        mostrar_status(vidas, pontos, sequencia)
        resultado, usou_dica = jogar_rodada(
            palavra, resposta, nome_idioma, codigo_fala, nivel["segundos"])

        if resultado == "sair":
            break
        rodadas += 1

        if resultado == "acertou":
            acertos += 1
            ganho = nivel["pontos"]
            if usou_dica:
                ganho = ganho // 2
            bonus = 5 * sequencia
            pontos += ganho + bonus
            sequencia += 1
            print(f"  ➕ +{ganho} pontos", end="")
            print(f"  🔥 bônus +{bonus}" if bonus else "")
        elif resultado == "errou":
            vidas -= 1
            sequencia = 0
            print("  💔 Você perdeu uma vida!")
        else:  # pulou
            sequencia = 0

    # ---------------- FIM DE JOGO ----------------
    print("\n  " + "═" * 44)
    if vidas <= 0:
        print("  💀 GAME OVER — suas vidas acabaram!")
    elif rodadas == len(palavras):
        print("  🏆 VITÓRIA! Você completou todas as palavras!")
    else:
        print("  👋 Jogo encerrado.")
    print("  " + "═" * 44)

    estrelas, comentario = classificar(acertos, rodadas)
    print(f"  📝 Acertos: {acertos}/{rodadas}")
    print(f"  🎯 Pontuação final: {pontos}")
    print(f"  {estrelas}  {comentario}")

    ranking = salvar_no_ranking(nome, pontos, nivel["nome"])
    mostrar_ranking(ranking)


# ============================================================
# ⚔️ MODO DISPUTA (2 a 4 jogadores no mesmo computador)
# ============================================================
def pedir_numero(pergunta, minimo, maximo, padrao):
    """Pergunta um número dentro de um intervalo (ENTER = padrão)."""
    while True:
        resposta = input(pergunta).strip()
        if resposta == "":
            return padrao
        if resposta.isdigit() and minimo <= int(resposta) <= maximo:
            return int(resposta)
        print(f"  Digite um número de {minimo} a {maximo}.")


def mostrar_placar(jogadores):
    print("\n  📊 PLACAR")
    print("  " + "─" * 36)
    ordenados = sorted(jogadores, key=lambda j: j["pontos"], reverse=True)
    for j in ordenados:
        fogo = f"🔥{j['sequencia']}" if j["sequencia"] >= 2 else ""
        print(f"  {j['nome']:<12} {j['pontos']:>4} pts  {fogo}")


def mostrar_vencedor(jogadores, turnos_jogados):
    print("\n  " + "═" * 44)
    print("  🏁 FIM DA DISPUTA!")
    print("  " + "═" * 44)

    ordenados = sorted(jogadores, key=lambda j: j["pontos"], reverse=True)
    posicoes = ["🥇", "🥈", "🥉", "4º"]
    for i, j in enumerate(ordenados):
        print(f"  {posicoes[i]} {j['nome']:<12} {j['pontos']:>4} pts"
              f"   ({j['acertos']} acertos)")

    maior = ordenados[0]["pontos"]
    campeoes = [j["nome"] for j in ordenados if j["pontos"] == maior]

    print()
    if turnos_jogados == 0 or maior == 0:
        print("  😅 Ninguém pontuou. Sem vencedor desta vez!")
    elif len(campeoes) == 1:
        print(f"  👑 {campeoes[0].upper()} VENCEU A DISPUTA! 🎉")
    else:
        print(f"  🤝 EMPATE entre: {' e '.join(campeoes)}!")


def disputa():
    print(r"""
  ╔══════════════════════════════════════════════╗
  ║   ⚔️   MODO DISPUTA  -  ATÉ 4 JOGADORES  ⚔️   ║
  ╠══════════════════════════════════════════════╣
  ║  Cada jogador responde na sua vez.           ║
  ║  Quem fizer mais pontos vence! 👑            ║
  ╚══════════════════════════════════════════════╝
    """)

    quantidade = pedir_numero("  👥 Quantos jogadores? (2-4, ENTER = 2): ", 2, 4, 2)

    jogadores = []
    for i in range(1, quantidade + 1):
        nome = input(f"  👤 Nome do jogador {i}: ").strip() or f"Jogador {i}"
        jogadores.append({"nome": nome, "pontos": 0,
                          "sequencia": 0, "acertos": 0})

    nivel = escolher_nivel()

    # Cada turno usa uma palavra diferente, para ninguém ver a resposta
    # que outro jogador errou.
    palavras = nivel["palavras"][:]
    random.shuffle(palavras)
    maximo_rodadas = len(palavras) // quantidade
    total_rodadas = pedir_numero(
        f"\n  🔁 Quantas rodadas? (1-{maximo_rodadas}, ENTER = {min(3, maximo_rodadas)}): ",
        1, maximo_rodadas, min(3, maximo_rodadas))

    print("\n  📖 REGRAS DA DISPUTA")
    print("   • Todos jogam em sequência, cada um com uma palavra diferente")
    print("   • Não há vidas: errar só zera a sua sequência 🔥")
    print("   • ENTER = falar | d = dica | p = pular | s = encerrar a disputa\n")
    calibrar_microfone()

    turnos_jogados = 0
    encerrou = False

    for rodada in range(1, total_rodadas + 1):
        print("\n  " + "═" * 44)
        print(f"  🏁 RODADA {rodada}/{total_rodadas}")
        print("  " + "═" * 44)

        for jogador in jogadores:
            palavra = palavras.pop()
            try:
                resposta = traduzir(palavra, codigo_idioma)
            except Exception as erro:
                print(f"  ⚠️  Não consegui traduzir '{palavra}': {erro}")
                continue

            print(f"\n  👉 Vez de {jogador['nome'].upper()}!"
                  f"  ⭐ {jogador['pontos']} pts")
            input("  (Passe o teclado/microfone e aperte ENTER quando estiver pronto) ")

            resultado, usou_dica = jogar_rodada(
                palavra, resposta, nome_idioma, codigo_fala, nivel["segundos"])

            if resultado == "sair":
                encerrou = True
                break
            turnos_jogados += 1

            if resultado == "acertou":
                ganho = nivel["pontos"] // 2 if usou_dica else nivel["pontos"]
                bonus = 5 * jogador["sequencia"]
                jogador["pontos"] += ganho + bonus
                jogador["sequencia"] += 1
                jogador["acertos"] += 1
                print(f"  ➕ {jogador['nome']} ganhou +{ganho} pontos", end="")
                print(f"  🔥 bônus +{bonus}" if bonus else "")
            else:  # errou ou pulou
                jogador["sequencia"] = 0

        if encerrou:
            print("\n  ⏹️  Disputa encerrada antes do fim.")
            break
        mostrar_placar(jogadores)

    mostrar_vencedor(jogadores, turnos_jogados)


# ============================================================
# 🕹️ MENU DE COMANDOS
# ============================================================
def escolher_modo(usar_argumento):
    """
    Pergunta o comando. Também aceita:  python main.py disputa
    """
    if usar_argumento and len(sys.argv) > 1 and sys.argv[1].lower() == "disputa":
        return "disputa"

    print("\n  ⌨️  COMANDOS")
    print("   • ENTER   → jogar sozinho 🎮")
    print("   • disputa → começar uma disputa com até 4 jogadores ⚔️")
    comando = input("  Digite o comando: ").strip().lower()
    return "disputa" if comando == "disputa" else "solo"


if __name__ == "__main__":
    primeira_vez = True
    while True:
        try:
            modo = escolher_modo(primeira_vez)
            primeira_vez = False
            if modo == "disputa":
                disputa()
            else:
                jogar()
        except KeyboardInterrupt:
            print("\n\n  Até a próxima! 👋")
            break
        de_novo = input("\n  🔁 Voltar ao menu? (s/n): ").strip().lower()
        if de_novo != "s":
            print("  Obrigado por jogar! 🎮")
            break