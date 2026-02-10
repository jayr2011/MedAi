<template>
  <div class="flex" :class="isUser ? 'justify-end' : 'justify-start'">
    <div
      v-if="!isUser"
      class="shrink-0 w-8 h-8 rounded-full bg-emerald-500 flex items-center justify-center mr-2 mt-1"
    >
      <span class="text-white text-xs font-bold">M</span>
    </div>

    <div
      class="max-w-[80%] px-4 py-3 rounded-2xl text-sm leading-relaxed"
      :class="
        isUser
          ? 'bg-emerald-500 text-white rounded-br-sm'
          : 'bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 shadow-sm border border-gray-100 dark:border-gray-700 rounded-bl-sm'
      "
    >
      <div v-if="!isUser && !message.content && loading" class="flex gap-1.5 py-1 px-1">
        <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
        <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
        <span class="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
      </div>

      <div
        v-else
        v-html="formattedContent"
        class="prose prose-sm dark:prose-invert max-w-none wrap-anywhere prose-a:block prose-a:max-w-full prose-a:truncate sm:prose-a:inline sm:prose-a:break-all sm:prose-a:wrap-anywhere sm:prose-a:whitespace-normal sm:prose-a:overflow-visible sm:prose-a:text-clip"
      />

      <span
        class="block text-[10px] mt-1.5 opacity-60"
        :class="isUser ? 'text-right text-emerald-100' : 'text-gray-400 dark:text-gray-500'"
      >
        {{ formattedTime }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { marked } from 'marked'
import type { Message } from '@/types/chat'

const props = defineProps<{
  message: Message
  loading?: boolean
}>()

const isUser = computed(() => props.message.role === 'user')

const formattedContent = computed(() => {
  if (!props.message.content) return ''
  return marked.parse(props.message.content, { breaks: true })
})

const formattedTime = computed(() =>
  props.message.timestamp.toLocaleTimeString('pt-BR', {
    hour: '2-digit',
    minute: '2-digit'
  })
)
</script>