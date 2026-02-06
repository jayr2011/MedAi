<template>
  <div class="p-3 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 transition-colors">
    <form @submit.prevent="handleSubmit" class="flex items-end gap-2">
      <textarea
        ref="inputRef"
        v-model="text"
        @keydown.enter.exact.prevent="handleSubmit"
        placeholder="Digite sua mensagem..."
        rows="1"
        class="flex-1 resize-none rounded-2xl border border-gray-200 dark:border-gray-600
               bg-gray-50 dark:bg-gray-700 text-gray-900 dark:text-gray-100
               placeholder-gray-400 dark:placeholder-gray-500
               px-4 py-3 text-sm focus:outline-none focus:ring-2
               focus:ring-emerald-500 focus:border-transparent
               max-h-32 overflow-y-auto transition-all scrollbar-none"
        :disabled="disabled"
      />
      <button
        type="submit"
        :disabled="!text.trim() || disabled"
        class="shrink-0 w-11 h-11 rounded-full bg-emerald-500 text-white flex items-center
               justify-center transition-all hover:bg-emerald-600 active:scale-95
               disabled:opacity-40 disabled:cursor-not-allowed disabled:active:scale-100"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
          <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
        </svg>
      </button>
    </form>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch } from 'vue'

defineProps<{ disabled?: boolean }>()
const emit = defineEmits<{ send: [text: string] }>()

const text = ref('')
const inputRef = ref<HTMLTextAreaElement>()

function handleSubmit() {
  if (!text.value.trim()) return
  emit('send', text.value.trim())
  text.value = ''
  nextTick(() => autoResize())
}

function autoResize() {
  const el = inputRef.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${el.scrollHeight}px`
}

watch(text, () => nextTick(() => autoResize()))
</script>