package service

import (
	"encoding/json"
	"fmt"
	"strings"
)

type ChatCompletionMessage struct {
	Role       string                   `json:"role"`
	Content    any                      `json:"content,omitempty"`
	Name       string                   `json:"name,omitempty"`
	ToolCalls  []ChatCompletionToolCall `json:"tool_calls,omitempty"`
	ToolCallID string                   `json:"tool_call_id,omitempty"`
}

type ChatCompletionToolCall struct {
	ID       string                     `json:"id"`
	Type     string                     `json:"type"`
	Function ChatCompletionFunctionCall `json:"function"`
}

type ChatCompletionFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ChatCompletionTool struct {
	Type     string                     `json:"type"`
	Function ChatCompletionToolFunction `json:"function"`
}

type ChatCompletionToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

type ChatCompletionRequest struct {
	Model               string                  `json:"model"`
	Messages            []ChatCompletionMessage `json:"messages"`
	MaxTokens           *int                    `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                    `json:"max_completion_tokens,omitempty"`
	Temperature         *float64                `json:"temperature,omitempty"`
	TopP                *float64                `json:"top_p,omitempty"`
	N                   *int                    `json:"n,omitempty"`
	Stream              *bool                   `json:"stream,omitempty"`
	Stop                any                     `json:"stop,omitempty"`
	PresencePenalty     *float64                `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64                `json:"frequency_penalty,omitempty"`
	LogitBias           map[string]float64      `json:"logit_bias,omitempty"`
	User                string                  `json:"user,omitempty"`
	Tools               []ChatCompletionTool    `json:"tools,omitempty"`
	ToolChoice          any                     `json:"tool_choice,omitempty"`
	Store               *bool                   `json:"store,omitempty"`
}

type ChatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []ChatCompletionChoice `json:"choices"`
	Usage   *ChatCompletionUsage   `json:"usage,omitempty"`
}

type ChatCompletionChoice struct {
	Index        int                   `json:"index"`
	Message      ChatCompletionMessage `json:"message"`
	FinishReason *string               `json:"finish_reason"`
	LogProbs     any                   `json:"logprobs,omitempty"`
}

type ChatCompletionUsage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

func ConvertChatCompletionToResponses(chatReq *ChatCompletionRequest) (map[string]any, error) {
	responsesReq := make(map[string]any)
	
	responsesReq["model"] = chatReq.Model
	
	if chatReq.Stream != nil {
		responsesReq["stream"] = *chatReq.Stream
	}
	
	if chatReq.Store != nil {
		responsesReq["store"] = *chatReq.Store
	}
	
	input := make([]map[string]any, 0, len(chatReq.Messages))
	for _, msg := range chatReq.Messages {
		inputItem := make(map[string]any)
		
		switch msg.Role {
		case "system":
			inputItem["type"] = "message"
			inputItem["role"] = "system"
			if content, ok := msg.Content.(string); ok && content != "" {
				inputItem["content"] = []map[string]any{
					{
						"type": "text",
						"text": content,
					},
				}
			}
			
		case "user":
			inputItem["type"] = "message"
			inputItem["role"] = "user"
			content := convertMessageContent(msg.Content)
			if content != nil {
				inputItem["content"] = content
			}
			
		case "assistant":
			if len(msg.ToolCalls) > 0 {
				for _, toolCall := range msg.ToolCalls {
					toolItem := make(map[string]any)
					toolItem["type"] = "function_call"
					toolItem["call_id"] = toolCall.ID
					toolItem["name"] = toolCall.Function.Name
					toolItem["arguments"] = toolCall.Function.Arguments
					input = append(input, toolItem)
				}
				continue
			}
			
			inputItem["type"] = "message"
			inputItem["role"] = "assistant"
			content := convertMessageContent(msg.Content)
			if content != nil {
				inputItem["content"] = content
			}
			
		case "tool":
			inputItem["type"] = "function_call_output"
			inputItem["call_id"] = msg.ToolCallID
			if content, ok := msg.Content.(string); ok {
				inputItem["output"] = content
			}
		}
		
		if len(inputItem) > 0 {
			input = append(input, inputItem)
		}
	}
	
	if len(input) > 0 {
		responsesReq["input"] = input
	}
	
	if chatReq.MaxTokens != nil {
		responsesReq["max_tokens"] = *chatReq.MaxTokens
	} else if chatReq.MaxCompletionTokens != nil {
		responsesReq["max_tokens"] = *chatReq.MaxCompletionTokens
	}
	
	if chatReq.Temperature != nil {
		responsesReq["temperature"] = *chatReq.Temperature
	}
	
	if chatReq.TopP != nil {
		responsesReq["top_p"] = *chatReq.TopP
	}
	
	if chatReq.PresencePenalty != nil {
		responsesReq["presence_penalty"] = *chatReq.PresencePenalty
	}
	
	if chatReq.FrequencyPenalty != nil {
		responsesReq["frequency_penalty"] = *chatReq.FrequencyPenalty
	}
	
	if chatReq.Stop != nil {
		responsesReq["stop"] = chatReq.Stop
	}
	
	if len(chatReq.Tools) > 0 {
		tools := make([]map[string]any, 0, len(chatReq.Tools))
		for _, tool := range chatReq.Tools {
			toolMap := map[string]any{
				"type": "function",
				"name": tool.Function.Name,
			}
			if tool.Function.Description != "" {
				toolMap["description"] = tool.Function.Description
			}
			if tool.Function.Parameters != nil {
				toolMap["parameters"] = tool.Function.Parameters
			}
			if tool.Function.Strict != nil {
				toolMap["strict"] = *tool.Function.Strict
			}
			tools = append(tools, toolMap)
		}
		responsesReq["tools"] = tools
	}
	
	if chatReq.ToolChoice != nil {
		responsesReq["tool_choice"] = chatReq.ToolChoice
	}
	
	return responsesReq, nil
}

func convertMessageContent(content any) any {
	if content == nil {
		return nil
	}
	
	if str, ok := content.(string); ok {
		return []map[string]any{
			{
				"type": "text",
				"text": str,
			},
		}
	}
	
	if arr, ok := content.([]any); ok {
		result := make([]map[string]any, 0, len(arr))
		for _, item := range arr {
			if itemMap, ok := item.(map[string]any); ok {
				result = append(result, itemMap)
			}
		}
		return result
	}
	
	return nil
}

func ConvertResponsesToChatCompletion(responsesResp map[string]any) (*ChatCompletionResponse, error) {
	chatResp := &ChatCompletionResponse{
		Object: "chat.completion",
	}
	
	if id, ok := responsesResp["id"].(string); ok {
		chatResp.ID = id
	}
	
	if created, ok := responsesResp["created"].(float64); ok {
		chatResp.Created = int64(created)
	}
	
	if model, ok := responsesResp["model"].(string); ok {
		chatResp.Model = model
	}
	
	if output, ok := responsesResp["output"].([]any); ok {
		choices := make([]ChatCompletionChoice, 0, 1)
		choice := ChatCompletionChoice{
			Index: 0,
			Message: ChatCompletionMessage{
				Role: "assistant",
			},
		}
		
		var contentParts []string
		var toolCalls []ChatCompletionToolCall
		
		for _, item := range output {
			if itemMap, ok := item.(map[string]any); ok {
				var itemType string
				if t, ok := itemMap["type"].(string); ok {
					itemType = t
				}
				
				switch itemType {
				case "message":
					if content, ok := itemMap["content"].([]any); ok {
						for _, c := range content {
							if cMap, ok := c.(map[string]any); ok {
								if cMap["type"] == "text" {
									if text, ok := cMap["text"].(string); ok {
										contentParts = append(contentParts, text)
									}
								}
							}
						}
					}
					
				case "function_call":
					toolCall := ChatCompletionToolCall{
						Type: "function",
					}
					if callID, ok := itemMap["call_id"].(string); ok {
						toolCall.ID = callID
					}
					if name, ok := itemMap["name"].(string); ok {
						toolCall.Function.Name = name
					}
					if args, ok := itemMap["arguments"].(string); ok {
						toolCall.Function.Arguments = args
					}
					toolCalls = append(toolCalls, toolCall)
				}
			}
		}
		
		if len(contentParts) > 0 {
			content := ""
			for _, part := range contentParts {
				content += part
			}
			choice.Message.Content = content
		}
		
		if len(toolCalls) > 0 {
			choice.Message.ToolCalls = toolCalls
		}
		
		if finishReason, ok := responsesResp["finish_reason"].(string); ok {
			choice.FinishReason = &finishReason
		}
		
		choices = append(choices, choice)
		chatResp.Choices = choices
	}
	
	if usage, ok := responsesResp["usage"].(map[string]any); ok {
		chatResp.Usage = &ChatCompletionUsage{}
		if inputTokens, ok := usage["input_tokens"].(float64); ok {
			chatResp.Usage.PromptTokens = int(inputTokens)
		}
		if outputTokens, ok := usage["output_tokens"].(float64); ok {
			chatResp.Usage.CompletionTokens = int(outputTokens)
		}
		chatResp.Usage.TotalTokens = chatResp.Usage.PromptTokens + chatResp.Usage.CompletionTokens
		
		if cacheRead, ok := usage["cache_read_input_tokens"].(float64); ok && cacheRead > 0 {
			chatResp.Usage.PromptTokensDetails = &PromptTokensDetails{
				CachedTokens: int(cacheRead),
			}
		}
	}
	
	return chatResp, nil
}

func ConvertResponsesStreamToChatCompletionStream(line string) (string, error) {
	if line == "" || line == "data: [DONE]" {
		return line, nil
	}
	
	var jsonData string
	if strings.HasPrefix(line, "data: ") {
		jsonData = line[6:]
	} else if strings.HasPrefix(line, "data:") {
		jsonData = line[5:]
	} else {
		return line, nil
	}
	
	var responsesChunk map[string]any
	if err := json.Unmarshal([]byte(jsonData), &responsesChunk); err != nil {
		return line, nil
	}
	
	chatChunk := map[string]any{
		"object": "chat.completion.chunk",
	}
	
	if id, ok := responsesChunk["id"].(string); ok {
		chatChunk["id"] = id
	}
	if created, ok := responsesChunk["created"].(float64); ok {
		chatChunk["created"] = int64(created)
	}
	if model, ok := responsesChunk["model"].(string); ok {
		chatChunk["model"] = model
	}
	
	choices := make([]map[string]any, 0, 1)
	choice := map[string]any{
		"index": 0,
		"delta": map[string]any{},
	}
	
	delta, _ := choice["delta"].(map[string]any)
	
	if output, ok := responsesChunk["output"].([]any); ok {
		for _, item := range output {
			if itemMap, ok := item.(map[string]any); ok {
				var itemType string
				if t, ok := itemMap["type"].(string); ok {
					itemType = t
				}
				
				switch itemType {
				case "message":
					if role, ok := itemMap["role"].(string); ok {
						delta["role"] = role
					}
					if content, ok := itemMap["content"].([]any); ok {
						for _, c := range content {
							if cMap, ok := c.(map[string]any); ok {
								if cMap["type"] == "text" {
									if text, ok := cMap["text"].(string); ok {
										delta["content"] = text
									}
								}
							}
						}
					}
					
				case "function_call":
					toolCalls := make([]map[string]any, 0, 1)
					toolCall := map[string]any{
						"index": 0,
						"type":  "function",
						"function": map[string]any{},
					}
					
					if callID, ok := itemMap["call_id"].(string); ok {
						toolCall["id"] = callID
					}
					if name, ok := itemMap["name"].(string); ok {
						if fn, ok := toolCall["function"].(map[string]any); ok {
							fn["name"] = name
						}
					}
					if args, ok := itemMap["arguments"].(string); ok {
						if fn, ok := toolCall["function"].(map[string]any); ok {
							fn["arguments"] = args
						}
					}
					
					toolCalls = append(toolCalls, toolCall)
					delta["tool_calls"] = toolCalls
				}
			}
		}
	}
	
	if finishReason, ok := responsesChunk["finish_reason"].(string); ok {
		choice["finish_reason"] = finishReason
	} else {
		choice["finish_reason"] = nil
	}
	
	choices = append(choices, choice)
	chatChunk["choices"] = choices
	
	if usage, ok := responsesChunk["usage"].(map[string]any); ok {
		chatUsage := map[string]any{}
		if inputTokens, ok := usage["input_tokens"].(float64); ok {
			chatUsage["prompt_tokens"] = int(inputTokens)
		}
		if outputTokens, ok := usage["output_tokens"].(float64); ok {
			chatUsage["completion_tokens"] = int(outputTokens)
		}
		if pt, ok := chatUsage["prompt_tokens"].(int); ok {
			if ct, ok := chatUsage["completion_tokens"].(int); ok {
				chatUsage["total_tokens"] = pt + ct
			}
		}
		chatChunk["usage"] = chatUsage
	}
	
	jsonBytes, err := json.Marshal(chatChunk)
	if err != nil {
		return "", fmt.Errorf("marshal chat completion chunk: %w", err)
	}
	
	return "data: " + string(jsonBytes), nil
}
