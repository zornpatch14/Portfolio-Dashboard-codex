FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install || true

COPY . /app

EXPOSE 3000

CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0", "--port", "3000"]
