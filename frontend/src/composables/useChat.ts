import { ref } from 'vue'
import type { Message } from '@/types/chat'

const API_URL = '/v1/chat/stream'

export function useChat() {
  const messages = ref<Message[]>([])
  const isLoading = ref(false)

  function createMessage(role: 'user' | 'assistant', content = ''): Message {
    return {
      id: Date.now().toString(36) + Math.random().toString(36).slice(2),
      role,
      content,
      timestamp: new Date()
    }
  }

  async function sendMessage(text: string) {
    if (!text.trim() || isLoading.value) return

    messages.value.push(createMessage('user', text))
    const assistantMsg = createMessage('assistant')
    messages.value.push(assistantMsg)
    isLoading.value = true

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [
            { role: 'system', content: 'Você é a MedIA, uma assistente médica virtual baseada no modelo Gemma. Sempre responda em português brasileiro. você lida com médicos registrados'},
            ...messages.value
              .filter(m => m.content)
              .map(m => ({ role: m.role, content: m.content })),
            { role: 'user', content: text }
          ]
        })
      })

      if (!response.ok) throw new Error(`Erro ${response.status}`)
      if (!response.body) throw new Error('Stream indisponível')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6).trim()
          if (data === '[DONE]') break

          try {
            const parsed = JSON.parse(data)
            const token = parsed.choices?.[0]?.delta?.content
            if (token) {
              const idx = messages.value.findIndex(m => m.id === assistantMsg.id)
              const msg = messages.value[idx]
              if (msg) msg.content += token
            }
          } catch {
          }
        }
      }
    } catch (error) {
      const idx = messages.value.findIndex(m => m.id === assistantMsg.id)
      const msg = messages.value[idx]
      if (msg) msg.content = 'Desculpe, ocorreu um erro. Tente novamente.'
      console.error(error)
    } finally {
      isLoading.value = false
    }
  }

  function clearMessages() {
    messages.value = []
  }

  return { messages, isLoading, sendMessage, clearMessages }
}