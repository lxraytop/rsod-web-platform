// router/index.js
import { createRouter, createWebHistory } from "vue-router";
import inference from "../views/Inference.vue"; // Yolo 推理验证页面

// 路由配置
const routes = [
  {
    path: "/",
    name: "inference",
    component: inference, // 默认打开就是检测页面
  },
];

// 创建路由实例
const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
