# build environment
FROM node:13.12.0-alpine as build
WORKDIR /app
ENV PATH /app/node_modules/.bin:$PATH
COPY package.json ./
COPY yarn.lock ./
RUN yarn --silent
RUN yarn add react-scripts@3.4.1 -g --silent
COPY . ./
RUN yarn build

# production environment
FROM nginx:stable-alpine
COPY --from=build /app/build /usr/share/nginx/html
ENV PORT 8080
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
