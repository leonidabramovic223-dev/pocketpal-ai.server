import type {RNLlamaOAICompatibleMessage} from 'llama.rn';

import {
  ApiCompletionParams,
  CompletionParams,
  CompletionResult,
  toApiCompletionParams,
} from '../../utils/completionTypes';

export const DEFAULT_MAX_TOKENS = 1024;
export const MAX_TOKENS_LIMIT = 2048;

export interface WireError {
  status: number;
  message: string;
  type: string;
  code: string;
}

export interface ValidatedChat {
  messages: RNLlamaOAICompatibleMessage[];
  nPredict: number;
  stop: string[] | undefined;
  sampling: Partial<CompletionParams>;
  responseFormat: CompletionParams['response_format'] | undefined;
  enableThinking: boolean | undefined;
  tools: CompletionParams['tools'] | undefined;
  toolChoice: CompletionParams['tool_choice'] | undefined;
  stream: boolean;
  includeUsage: boolean;
}

export type ValidatedChatRequest =
  | {ok: true; request: ValidatedChat}
  | {ok: false; error: WireError};

type Json = Record<string, unknown>;

const ROLES = new Set(['system', 'user', 'assistant', 'tool']);

const SAMPLING_FIELDS: [string, keyof CompletionParams][] = [
  ['temperature', 'temperature'],
  ['top_p', 'top_p'],
  ['top_k', 'top_k'],
  ['min_p', 'min_p'],
  ['seed', 'seed'],
  ['frequency_penalty', 'penalty_freq'],
  ['presence_penalty', 'penalty_present'],
  ['repeat_penalty', 'penalty_repeat'],
];

class RequestError extends Error {
  constructor(
    message: string,
    readonly code: 'invalid_request' | 'unsupported_parameter',
  ) {
    super(message);
  }
}

const isObject = (value: unknown): value is Json =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isPresent = (value: unknown) => value !== undefined && value !== null;

function unsupported(name: string): never {
  throw new RequestError(`${name} is not supported`, 'unsupported_parameter');
}

function invalid(message: string): never {
  throw new RequestError(message, 'invalid_request');
}

function rejectUnsupported(body: Json) {
  if (isPresent(body.n) && body.n !== 1) {
    unsupported('n');
  }
  if (body.logprobs === true) {
    unsupported('logprobs');
  }
}

function messageContent(content: unknown): string {
  if (typeof content === 'string') {
    return content;
  }
  if (!Array.isArray(content)) {
    invalid('message content must be a string or an array of parts');
  }
  return content
    .map(part => {
      if (!isObject(part) || part.type !== 'text') {
        unsupported('non-text content parts');
      }
      if (typeof part.text !== 'string') {
        invalid('text parts need a string text');
      }
      return part.text;
    })
    .join('');
}

function toWireToolCalls(value: unknown) {
  if (!Array.isArray(value)) {
    invalid('tool_calls must be an array');
  }
  return value.map((call, index) => {
    if (!isObject(call) || !isObject(call.function)) {
      invalid('each tool call needs a function');
    }
    const fn = call.function;
    if (typeof fn.name !== 'string' || fn.name === '') {
      invalid('each tool call needs a function name');
    }
    const args =
      typeof fn.arguments === 'string'
        ? fn.arguments
        : JSON.stringify(fn.arguments ?? {});
    return {
      id:
        typeof call.id === 'string' && call.id !== ''
          ? call.id
          : `call_${index}`,
      type: 'function',
      function: {name: fn.name, arguments: args},
    };
  });
}

function toMessages(messages: unknown): RNLlamaOAICompatibleMessage[] {
  if (!Array.isArray(messages) || messages.length === 0) {
    invalid('messages must be a non-empty array');
  }
  return messages.map(message => {
    if (!isObject(message) || !ROLES.has(message.role as string)) {
      invalid('each message needs a role of system, user, assistant or tool');
    }
    const role = message.role as string;
    const out: Json = {role};
    // An assistant turn that only called tools carries null content.
    const calls =
      role === 'assistant' && isPresent(message.tool_calls)
        ? toWireToolCalls(message.tool_calls)
        : undefined;
    out.content =
      calls && !isPresent(message.content)
        ? ''
        : messageContent(message.content);
    if (calls && calls.length > 0) {
      out.tool_calls = calls;
    }
    if (role === 'tool') {
      if (typeof message.tool_call_id !== 'string' || !message.tool_call_id) {
        invalid('tool messages need a tool_call_id');
      }
      out.tool_call_id = message.tool_call_id;
      if (typeof message.name === 'string') {
        out.name = message.name;
      }
    }
    return out as unknown as RNLlamaOAICompatibleMessage;
  });
}

function toTools(value: unknown): CompletionParams['tools'] | undefined {
  if (!isPresent(value)) {
    return undefined;
  }
  if (!Array.isArray(value)) {
    invalid('tools must be an array');
  }
  if (value.length === 0) {
    return undefined;
  }
  for (const tool of value) {
    if (
      !isObject(tool) ||
      tool.type !== 'function' ||
      !isObject(tool.function) ||
      typeof tool.function.name !== 'string' ||
      tool.function.name === ''
    ) {
      invalid('each tool must be {type: "function", function: {name, ...}}');
    }
  }
  return value as CompletionParams['tools'];
}

function toToolChoice(
  value: unknown,
  hasTools: boolean,
): CompletionParams['tool_choice'] | undefined {
  if (!isPresent(value)) {
    return undefined;
  }
  if (!hasTools) {
    return undefined;
  }
  if (typeof value === 'string') {
    if (!['auto', 'none', 'required'].includes(value)) {
      invalid('tool_choice must be auto, none, required or a function');
    }
    return value;
  }
  if (
    isObject(value) &&
    value.type === 'function' &&
    isObject(value.function) &&
    typeof value.function.name === 'string'
  ) {
    // llama.cpp parses only auto/none/required; a named function is the
    // closest to "required". The model may pick another tool.
    return 'required';
  }
  invalid('tool_choice must be auto, none, required or a function');
}

function finiteOrAbsent(body: Json, name: string): number | undefined {
  const value = body[name];
  if (!isPresent(value)) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    invalid(`${name} must be a number`);
  }
  return value;
}

function maxTokens(body: Json): number {
  const requested =
    finiteOrAbsent(body, 'max_completion_tokens') ??
    finiteOrAbsent(body, 'max_tokens');
  if (requested === undefined || requested <= 0) {
    return DEFAULT_MAX_TOKENS;
  }
  return Math.min(MAX_TOKENS_LIMIT, Math.max(1, Math.floor(requested)));
}

function clientStop(stop: unknown): string[] | undefined {
  if (!isPresent(stop)) {
    return undefined;
  }
  const requested = typeof stop === 'string' ? [stop] : stop;
  if (
    !Array.isArray(requested) ||
    requested.some(word => typeof word !== 'string')
  ) {
    invalid('stop must be a string or an array of strings');
  }
  return requested as string[];
}

function sampling(body: Json): Partial<CompletionParams> {
  const fields: Json = {};
  for (const [name, key] of SAMPLING_FIELDS) {
    const value = finiteOrAbsent(body, name);
    if (value !== undefined) {
      fields[key] = value;
    }
  }
  return fields as Partial<CompletionParams>;
}

function requestedThinking(body: Json): boolean | undefined {
  const kwargs = body.chat_template_kwargs;
  if (isObject(kwargs) && typeof kwargs.enable_thinking === 'boolean') {
    return kwargs.enable_thinking;
  }
  return undefined;
}

/**
 * Checks an OpenAI chat-completion body without reading any app state. Fields
 * the server cannot honour are rejected, not dropped, so a client never
 * mistakes an ignored tool or image for a model that declined it.
 */
export function validateChatRequest(body: unknown): ValidatedChatRequest {
  try {
    if (!isObject(body)) {
      invalid('request body must be a JSON object');
    }
    rejectUnsupported(body);
    const streamOptions = body.stream_options;
    const tools = toTools(body.tools);
    return {
      ok: true,
      request: {
        tools,
        toolChoice: toToolChoice(body.tool_choice, tools !== undefined),
        messages: toMessages(body.messages),
        nPredict: maxTokens(body),
        stop: clientStop(body.stop),
        sampling: sampling(body),
        responseFormat: isObject(body.response_format)
          ? (body.response_format as CompletionParams['response_format'])
          : undefined,
        enableThinking: requestedThinking(body),
        stream: body.stream === true,
        includeUsage:
          isObject(streamOptions) && streamOptions.include_usage === true,
      },
    };
  } catch (error) {
    if (error instanceof RequestError) {
      return {
        ok: false,
        error: {
          status: 400,
          message: error.message,
          type: 'invalid_request_error',
          code: error.code,
        },
      };
    }
    throw error;
  }
}

export function buildChatParams(
  chat: ValidatedChat,
  base: CompletionParams,
  stopWords: string[] = [],
): ApiCompletionParams {
  const params: CompletionParams = {
    ...base,
    messages: chat.messages,
    n_predict: chat.nPredict,
    stop: chat.stop ? [...new Set([...stopWords, ...chat.stop])] : stopWords,
    ...chat.sampling,
  };
  if (chat.responseFormat) {
    params.response_format = chat.responseFormat;
  }
  if (chat.tools) {
    params.tools = chat.tools;
    params.tool_choice = chat.toolChoice ?? 'auto';
  }
  const enableThinking = chat.enableThinking ?? base.enable_thinking;

  const apiParams = toApiCompletionParams(params);
  apiParams.reasoning_format = 'auto';
  if (enableThinking !== undefined) {
    apiParams.enable_thinking = enableThinking;
    apiParams.chat_template_kwargs = {
      ...apiParams.chat_template_kwargs,
      enable_thinking: enableThinking,
    };
  }
  return apiParams;
}

export interface CompletionIds {
  id: string;
  /** Unix seconds at request start. */
  created: number;
  model: string;
}

export function newCompletionIds(
  model: string,
  now: number = Date.now(),
): CompletionIds {
  const random = Math.random().toString(36).slice(2, 14);
  return {id: `chatcmpl-${random}`, created: Math.floor(now / 1000), model};
}

export function finishReason(
  result: CompletionResult,
): 'length' | 'stop' | 'tool_calls' {
  if (result.tool_calls && result.tool_calls.length > 0) {
    return 'tool_calls';
  }
  return result.stopped_limit || result.context_full ? 'length' : 'stop';
}

/** OpenAI-shaped tool calls, with a stable id when llama.rn returns none. */
export function wireToolCalls(
  result: CompletionResult,
  ids: CompletionIds,
): WireToolCall[] {
  return (result.tool_calls ?? []).map((call, index) => ({
    id: call.id || `call_${ids.id.slice(-8)}_${index}`,
    type: 'function',
    function: {
      name: call.function?.name ?? '',
      arguments: call.function?.arguments ?? '',
    },
  }));
}

export interface WireToolCall {
  id: string;
  type: 'function';
  function: {name: string; arguments: string};
}

export function usageOf(result: CompletionResult) {
  const promptTokens = result.tokens_evaluated ?? 0;
  const completionTokens = result.tokens_predicted ?? 0;
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  };
}

export function toCompletion(result: CompletionResult, ids: CompletionIds) {
  const toolCalls = wireToolCalls(result, ids);
  // result.text is the raw model output, markup included. When llama.rn has
  // already split the turn into reasoning and tool calls, falling back to it
  // would repeat them as literal text in content.
  const parsedAway = Boolean(result.reasoning_content) || toolCalls.length > 0;
  const content = result.content ?? (parsedAway ? '' : result.text);
  return {
    id: ids.id,
    object: 'chat.completion',
    created: ids.created,
    model: ids.model,
    choices: [
      {
        index: 0,
        message: {
          role: 'assistant',
          content: toolCalls.length > 0 && !content ? null : content,
          ...(result.reasoning_content
            ? {reasoning_content: result.reasoning_content}
            : {}),
          ...(toolCalls.length > 0 ? {tool_calls: toolCalls} : {}),
        },
        finish_reason: finishReason(result),
      },
    ],
    usage: usageOf(result),
  };
}

export interface ChunkDelta {
  role?: 'assistant';
  content?: string;
  reasoning_content?: string;
  tool_calls?: (WireToolCall & {index: number})[];
}

export function toChunk(delta: ChunkDelta, ids: CompletionIds) {
  return {
    id: ids.id,
    object: 'chat.completion.chunk',
    created: ids.created,
    model: ids.model,
    choices: [{index: 0, delta, finish_reason: null}],
  };
}

export function toFinalChunk(
  result: CompletionResult,
  ids: CompletionIds,
  includeUsage: boolean,
) {
  return {
    id: ids.id,
    object: 'chat.completion.chunk',
    created: ids.created,
    model: ids.model,
    choices: [{index: 0, delta: {}, finish_reason: finishReason(result)}],
    ...(includeUsage ? {usage: usageOf(result)} : {}),
  };
}

export function errorBody(error: Omit<WireError, 'status'>) {
  return {
    error: {message: error.message, type: error.type, code: error.code},
  };
}
