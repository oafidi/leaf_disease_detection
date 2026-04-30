#!/bin/bash
# =============================================================================
# GUIDE DE DÉPLOIEMENT APPLESCAN SUR AWS EC2
# Lire et exécuter BLOC PAR BLOC — ne pas tout lancer d'un coup.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 0 — PRÉREQUIS (sur ta machine locale)
# ─────────────────────────────────────────────────────────────────────────────

# 0a. Avoir un compte AWS et AWS CLI configuré
#     https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
#     aws configure  →  entre ta Access Key / Secret / région (ex: eu-west-3)

# 0b. Ton domaine doit pointer vers l'IP de ta future instance.
#     Tu feras ça à l'étape 3 (après avoir l'IP de l'instance).

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — CRÉER L'INSTANCE EC2 (via Console AWS ou CLI)
# ─────────────────────────────────────────────────────────────────────────────

# Type recommandé : t3.medium (2 vCPU, 4 GB RAM) — le modèle ML est lourd.
# AMI : Ubuntu 24.04 LTS (chercher "ubuntu-24.04" dans la console)
# Stockage : 20 GB minimum (SSD gp3)

# Via AWS CLI :
aws ec2 run-instances \
  --image-id ami-0a0d8591ef83328bf \   # Ubuntu 24.04 eu-west-3 (Paris) — vérifier l'AMI ID de ta région
  --instance-type t3.medium \
  --key-name MA_CLE_SSH \              # Nom de ta paire de clés SSH AWS
  --security-group-ids sg-XXXXXXXX \  # Voir étape 1b
  --count 1 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=applescan}]'

# 1b. Security Group — ouvrir uniquement les ports nécessaires :
#
#   Port 22  (SSH)   — ton IP uniquement (pas 0.0.0.0/0 !)
#   Port 80  (HTTP)  — 0.0.0.0/0  (pour Let's Encrypt + redirection vers HTTPS)
#   Port 443 (HTTPS) — 0.0.0.0/0
#
# Via console : EC2 > Security Groups > Inbound rules > Add rule

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — IP ÉLASTIQUE (pour ne pas changer d'IP au redémarrage)
# ─────────────────────────────────────────────────────────────────────────────

# Dans la console AWS : EC2 > Elastic IPs > Allocate > Associate to instance
# Ou CLI :
aws ec2 allocate-address --domain vpc
# Récupérer l'AllocationId retourné, puis :
aws ec2 associate-address --instance-id i-XXXXXXXXXXXXXXXXX --allocation-id eipalloc-XXXXXXXXXXXXXXXXX

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — CONFIGURER LE DNS (chez ton registrar de domaine)
# ─────────────────────────────────────────────────────────────────────────────

# Aller dans le panneau DNS de ton registrar (OVH, Namecheap, Cloudflare, etc.)
# Ajouter ces deux enregistrements :
#
#   Type  Nom    Valeur                  TTL
#   A     @      TON_IP_ELASTIQUE        300
#   A     www    TON_IP_ELASTIQUE        300
#
# Attendre la propagation DNS (5-30 min). Vérifier avec :
#   dig TON_DOMAINE.COM +short
# → Doit retourner ton IP élastique

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — SE CONNECTER AU SERVEUR ET INSTALLER DOCKER
# ─────────────────────────────────────────────────────────────────────────────

# Connexion SSH :
# ssh -i ~/.ssh/MA_CLE_SSH.pem ubuntu@TON_IP_ELASTIQUE

# Sur le serveur — installer Docker :
sudo apt update && sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Ajouter l'utilisateur ubuntu au groupe docker (évite sudo à chaque fois)
sudo usermod -aG docker ubuntu
newgrp docker

# Vérifier :
docker --version
docker compose version

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — UPLOADER LE PROJET SUR LE SERVEUR
# ─────────────────────────────────────────────────────────────────────────────

# Depuis ta machine locale (dans le dossier du projet) :
# scp -i ~/.ssh/MA_CLE_SSH.pem -r apple_leaf_disease_detection/ ubuntu@TON_IP_ELASTIQUE:~/

# Le modèle .keras est lourd — le transférer séparément si nécessaire :
# scp -i ~/.ssh/MA_CLE_SSH.pem vision-service/apple_leaf_model_final.keras \
#     ubuntu@TON_IP_ELASTIQUE:~/apple_leaf_disease_detection/vision-service/

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 — CONFIGURER L'ENVIRONNEMENT SUR LE SERVEUR
# ─────────────────────────────────────────────────────────────────────────────

# Sur le serveur :
cd ~/apple_leaf_disease_detection

# Remplir le .env avec les vraies valeurs
nano .env
# → Remplacer tous les CHANGE_ME_...
# → Mettre ta vraie clé OpenAI
# → Mettre ton vrai domaine dans ALLOWED_ORIGINS

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 7 — DÉMARRER EN HTTP POUR OBTENIR LE CERTIFICAT SSL
# ─────────────────────────────────────────────────────────────────────────────

# Créer les dossiers certbot
mkdir -p nginx/certbot/conf nginx/certbot/www

# Remplacer TON_DOMAINE.COM dans la config Nginx HTTP
sed -i 's/TON_DOMAINE.COM/votredomaine.com/g' nginx/conf.d/applescan-http.conf

# Démarrer uniquement Nginx (en mode HTTP, pour que certbot puisse répondre)
docker compose up -d nginx-proxy

# Vérifier que Nginx tourne :
docker compose ps nginx-proxy

# Obtenir le certificat SSL (remplacer les valeurs) :
docker compose run --rm certbot certonly \
  --webroot \
  --webroot-path /var/www/certbot \
  --email ton@email.com \
  --agree-tos \
  --no-eff-email \
  -d votredomaine.com \
  -d www.votredomaine.com

# Si succès → le certificat est dans nginx/certbot/conf/live/votredomaine.com/

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 8 — ACTIVER LA CONFIG HTTPS
# ─────────────────────────────────────────────────────────────────────────────

# Remplacer TON_DOMAINE.COM dans la config HTTPS
sed -i 's/TON_DOMAINE.COM/votredomaine.com/g' nginx/conf.d/applescan-https.conf

# Basculer vers la config HTTPS :
mv nginx/conf.d/applescan-http.conf nginx/conf.d/applescan-http.conf.bak
cp nginx/conf.d/applescan-https.conf nginx/conf.d/applescan.conf

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 9 — DÉMARRER TOUS LES SERVICES
# ─────────────────────────────────────────────────────────────────────────────

docker compose up -d --build

# Vérifier que tout tourne :
docker compose ps

# Voir les logs en temps réel :
docker compose logs -f

# Tester l'accès :
#   https://votredomaine.com          → frontend
#   https://votredomaine.com/api/auth/health  → {"status":"ok"}
#   https://votredomaine.com/api/vision/health → {"status":"ok","model_loaded":true}

# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 10 — COMMANDES UTILES AU QUOTIDIEN
# ─────────────────────────────────────────────────────────────────────────────

# Voir l'état des conteneurs :
docker compose ps

# Redémarrer un service après modif :
docker compose up -d --build auth-service

# Voir les logs d'un service :
docker compose logs -f chat-service

# Recharger Nginx sans coupure :
docker compose exec nginx-proxy nginx -s reload

# Arrêter tout (sans effacer les données) :
docker compose stop

# Forcer le renouvellement SSL manuel :
docker compose run --rm certbot renew --force-renewal
docker compose exec nginx-proxy nginx -s reload
