FROM node:22-bookworm-slim AS build
WORKDIR /app
COPY UI/package.json UI/package-lock.json ./
RUN npm ci
COPY UI/index.html UI/tsconfig.json UI/vite.config.ts ./
COPY UI/src ./src
RUN npm run build

FROM nginx:stable-alpine
COPY UI/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
