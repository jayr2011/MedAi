<template>
  <div class="flex flex-col h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
    <header class="shrink-0 bg-emerald-600 dark:bg-gray-800 text-white px-4 py-3 shadow-md h-14">
      <div class="flex items-center justify-between max-w-2xl mx-auto h-full">
        <div class="flex items-center gap-3 h-full">
          <img
            src="@/assets/medai_trans.png"
            alt="MedIA"
            class="h-26 invert brightness-0 dark:invert"
          />
          <p class="text-xs text-emerald-100 dark:text-gray-400">
            {{ isLoading ? 'Digitando...' : 'Assistente médico' }}
          </p>
        </div>

        <div class="flex items-center gap-2">
          <!-- Limpar conversa -->
          <button
            @click="handleClear"
            :disabled="messages.length === 0"
            class="w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            title="Limpar conversa"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 24 24" fill="none"
                stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
              <line x1="10" y1="11" x2="10" y2="17" />
              <line x1="14" y1="11" x2="14" y2="17" />
            </svg>
          </button>

          <!-- Dark mode -->
          <button
            @click="toggle"
            class="w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 flex items-center justify-center transition-colors"
          >
            <span v-if="isDark" class="text-lg">☀️</span>
            <span v-else class="text-lg">🌙</span>
          </button>
        </div>
      </div>
    </header>

    <div ref="scrollContainer" class="flex-1 overflow-y-auto relative" @scroll="handleScroll">
      <div class="max-w-2xl mx-auto px-4 py-4 space-y-4">
        <!-- Welcome -->
        <div class="w-16 h-16 rounded-full bg-emerald-100 flex items-center justify-center mb-4">
          <span class="text-2xl drop-shadow-md">🩺</span>
        </div>
        <h2 class="text-lg font-semibold text-gray-700 dark:text-gray-200">Olá! Eu sou a MedIA</h2>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1 max-w-xs">
          Sua assistente médica virtual. Como posso te ajudar hoje?
        </p>

        <ChatBubble
          v-for="msg in messages"
          :key="msg.id"
          :message="msg"
          :loading="isLoading && msg.role === 'assistant' && msg === messages[messages.length - 1]"
        />
      </div>

      <Transition name="fade">
        <button
          v-if="showScrollBtn"
          @click="scrollToBottom"
          class="sticky bottom-4 left-1/2 -translate-x-1/2 w-10 h-10 rounded-full
                 bg-emerald-500 text-white shadow-lg flex items-center justify-center
                 hover:bg-emerald-600 active:scale-95 transition-all"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 5v14M5 12l7 7 7-7" />
          </svg>
        </button>
      </Transition>
    </div>

    <div class="shrink-0 bg-white dark:bg-gray-800 px-4 py-4 border-t border-gray-200 dark:border-gray-700">
      <div class="max-w-2xl mx-auto w-full">
        <ChatInput :disabled="isLoading" @send="handleSend" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch } from 'vue'
import ChatBubble from '@/components/ChatBubble.vue'
import ChatInput from '@/components/ChatInput.vue'
import { useChat } from '../composables/useChat'
import { useDarkMode } from '../composables/useDark'

const { messages, isLoading, sendMessage, clearMessages } = useChat()
const { isDark, toggle } = useDarkMode()
const scrollContainer = ref<HTMLElement>()
const showScrollBtn = ref(false)

function handleSend(text: string) {
  sendMessage(text)
}

function handleClear() {
  if (confirm('Tem certeza que deseja limpar a conversa?')) {
    clearMessages()
  }
}

function handleScroll() {
  const el = scrollContainer.value
  if (!el) return
  const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight
  showScrollBtn.value = distanceFromBottom > 200
}

function scrollToBottom() {
  nextTick(() => {
    const el = scrollContainer.value
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  })
}

watch(messages, scrollToBottom, { deep: true })
</script>