/// <reference types="vite/client" />

import { ref } from 'vue'
import type { Message } from '@/types/chat'

interface ImportMetaEnv {
  readonly VITE_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

const API_URL = import.meta.env.VITE_API_URL || '/v1/chat/stream'

export function useChat() {
  const messages = ref<Message[]>([])
  const isLoading = ref(false)

  function createMessage(role: 'user' | 'assistant', content = ''): Message {
    return {
      id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      role,
      content,
      timestamp: new Date(),
    }
  }

  async function sendMessage(text: string): Promise<void> {
    if (!text?.trim() || isLoading.value) return

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
            {
              role: 'system',
              content:
                'Você é a MedIA, uma assistente médica. Sempre responda em português brasileiro. você lida com médicos registrados',
            },
            ...messages.value
              .filter((m) => m.content)
              .map((m) => ({ role: m.role, content: m.content })),
          ],
        }),
      })

      if (!response.ok) throw new Error(`Erro ${response.status}`)
      if (!response.body) throw new Error('Stream indisponível')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let receivedDone = false

      const processEvent = (eventBlock: string) => {
        const dataLines = eventBlock
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())

        if (dataLines.length === 0) return false

        const data = dataLines.join('\n').trim()
        if (!data) return false
        if (data === '[DONE]') {
          receivedDone = true
          return true
        }

        const parsed = JSON.parse(data)
        const errorMessage = parsed?.error?.message
        if (errorMessage) {
          throw new Error(errorMessage)
        }

        const token = parsed.choices?.[0]?.delta?.content
        if (token) {
          const idx = messages.value.findIndex((m) => m.id === assistantMsg.id)
          const msg = messages.value[idx]
          if (msg) msg.content += token
        }

        return false
      }

      while (true) {
        const { done, value } = await reader.read()
        const decoded = decoder.decode(value || new Uint8Array(), { stream: !done })
        buffer += decoded.replace(/\r\n/g, '\n').replace(/\r/g, '\n')

        let shouldStop = false
        let sepIndex = buffer.indexOf('\n\n')
        while (sepIndex !== -1) {
          const eventBlock = buffer.slice(0, sepIndex)
          buffer = buffer.slice(sepIndex + 2)
          shouldStop = processEvent(eventBlock)
          if (shouldStop) break
          sepIndex = buffer.indexOf('\n\n')
        }

        if (shouldStop) break
        if (done) {
          if (buffer.trim()) processEvent(buffer)
          if (!receivedDone) {
            throw new Error('Stream encerrado sem marcador [DONE].')
          }
          break
        }
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Erro desconhecido'
      const idx = messages.value.findIndex((m) => m.id === assistantMsg.id)
      const msg = messages.value[idx]
      if (msg) {
        if (msg.content.trim()) {
          msg.content +=
            '\n\n_Resposta interrompida antes do fim. Tente novamente ou envie "continue de onde parou"._'
        } else {
          msg.content = `Desculpe, ocorreu um erro: ${errorMessage}. Tente novamente.`
        }
      }
      console.error('Chat error:', error)
      throw error
    } finally {
      isLoading.value = false
    }
  }

  function clearMessages() {
    messages.value = []
  }

  return { messages, isLoading, sendMessage, clearMessages }
}
