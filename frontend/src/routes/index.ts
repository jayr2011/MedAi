import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      redirect: '/meAi'
    },
    {
      path: '/meAi',
      name: 'chat',
      component: () => import('@/views/ChatView.vue')
    }
  ]
})

export default router