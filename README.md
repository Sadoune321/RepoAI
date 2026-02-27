1️⃣ Préparer ton projet local

Structure des fichiers
Exemple pour skin-api :

mlops2/
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── deployment.yaml
└── README.md

main.py → ton API FastAPI

requirements.txt → dépendances Python

Dockerfile → construire l’image

docker-compose.yml → tester localement

deployment.yaml → déploiement Kubernetes

README.md → doc du projet

Initialiser Git

git init
git add .
git commit -m "Initial commit: API skeleton"
git branch -M main
git remote add origin https://github.com/Sadoune321/lung-colon-cancer-AI.git
git push -u origin main
2️⃣ Construire et tester Docker localement

Pointer vers le Docker de Minikube

eval $(minikube -p minikube docker-env)

Construire l’image

docker build -t skin-api:latest .

Tester l’image localement avec Docker

docker run -p 8000:8000 skin-api:latest

→ Vérifie que ton API fonctionne sur http://localhost:8000.

3️⃣ Déployer sur Kubernetes (Minikube)

Vérifier que Minikube fonctionne

minikube start
kubectl get nodes

Appliquer ton deployment.yaml

kubectl apply -f deployment.yaml

image: skin-api:latest → image locale

imagePullPolicy: IfNotPresent → pas besoin de Docker Hub

Créer un service NodePort (si pas déjà fait)

apiVersion: v1
kind: Service
metadata:
  name: skin-api-service
spec:
  selector:
    app: skin-api
  type: NodePort
  ports:
    - port: 8000
      targetPort: 8000
      nodePort: 32131
kubectl apply -f service.yaml

Vérifier les pods et service

kubectl get pods -w
kubectl get services
minikube service skin-api-service
4️⃣ Workflow pour modifications et mises à jour
A. Modifier le code

main.py ou deployment.yaml, etc.

B. Tester localement

Docker : docker build -t skin-api:latest .

Docker run : docker run -p 8000:8000 skin-api:latest

C. Mettre à jour Kubernetes

Rebuild image si Dockerfile ou code changé

Appliquer deployment.yaml

kubectl apply -f deployment.yaml
kubectl rollout status deployment/skin-api-deployment
D. Pusher les changements sur GitHub
git add .
git commit -m "Update API / fix deployment"
git push origin main
5️⃣ Gestion des images volumineuses

Ne jamais push les .pth ou images Docker sur GitHub.

Si besoin de partager modèle :

Utiliser Git LFS pour .pth (>100 MB)

Ou Docker Hub pour l’image entière
