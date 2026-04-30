#!/bin/bash
# =============================================================================
# setup-domain.sh — Remplace TON_DOMAINE.COM partout en une seule commande.
# Usage : bash deploy/setup-domain.sh votredomaine.com votre@email.com
# =============================================================================

DOMAIN=${1:-""}
EMAIL=${2:-""}

if [ -z "$DOMAIN" ] || [ -z "$EMAIL" ]; then
  echo "Usage : bash deploy/setup-domain.sh votredomaine.com votre@email.com"
  exit 1
fi

echo "→ Domaine : $DOMAIN"
echo "→ Email   : $EMAIL"
echo ""

# Remplacer dans les configs Nginx
sed -i "s/TON_DOMAINE.COM/$DOMAIN/g" nginx/conf.d/applescan-http.conf
sed -i "s/TON_DOMAINE.COM/$DOMAIN/g" nginx/conf.d/applescan-https.conf
echo "✓ nginx/conf.d/applescan-http.conf  mis à jour"
echo "✓ nginx/conf.d/applescan-https.conf mis à jour"

# Mettre à jour ALLOWED_ORIGINS dans .env
sed -i "s|ALLOWED_ORIGINS=.*|ALLOWED_ORIGINS=https://$DOMAIN,https://www.$DOMAIN|" .env
echo "✓ .env ALLOWED_ORIGINS mis à jour"

echo ""
echo "Prochaine étape — obtenir le certificat SSL :"
echo ""
echo "  docker compose up -d nginx-proxy"
echo "  docker compose run --rm certbot certonly \\"
echo "    --webroot --webroot-path /var/www/certbot \\"
echo "    --email $EMAIL --agree-tos --no-eff-email \\"
echo "    -d $DOMAIN -d www.$DOMAIN"
echo ""
echo "Ensuite activer HTTPS :"
echo "  mv nginx/conf.d/applescan-http.conf nginx/conf.d/applescan-http.conf.bak"
echo "  cp nginx/conf.d/applescan-https.conf nginx/conf.d/applescan.conf"
echo "  docker compose up -d --build"
