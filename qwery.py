from dotenv import load_dotenv  # type: ignore
load_dotenv()  # Load environment variables from .env file

import asyncio
import os
from typing import List, Dict, Optional
import pinecone  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
from google import genai  # type: ignore
from datetime import datetime
from mcp.client.stdio import stdio_client
from mcp import ClientSession, types
import sys
# ==================== Configuration ====================

MCP_SERVER_CMD = ["uvx", "mcp-postgresql-ops"] 
MCP_ENV = {
    "POSTGRES_HOST": os.getenv("POSTGRES_HOST"),
    "POSTGRES_PORT": os.getenv("POSTGRES_PORT"),
    "POSTGRES_DB": os.getenv("POSTGRES_DB"),
    "POSTGRES_USER": os.getenv("POSTGRES_USER"),
    "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD")
}
# Initialize embeddings client
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# Configure Gemini for text generation
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✓ Gemini API configured")
else:
    print("✗ Warning: GEMINI_API_KEY not found in .env file")
    client = None

# Initialize Pinecone client
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pinecone_index = None
if PINECONE_API_KEY and PINECONE_INDEX_NAME:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✓ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
else:
    print("✗ Pinecone not configured – set PINECONE_API_KEY and PINECONE_INDEX_NAME in .env")

# Conversation history for multi-turn conversations
conversation_history: List[Dict[str, str]] = []

# ==================== Helper Functions ====================

async def mcp_query(question: str) -> str:
    """Connects to mcp_server.py via stdio subprocess"""
    try:
        from mcp.client.stdio import stdio_client
        from mcp import ClientSession, StdioServerParameters

        # Define server parameters properly
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["mcp_server.py"],
            env={**os.environ, "PYTHONIOENCODING": "utf-8"}
            )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List available tools (optional, good for debugging)
                tools = await session.list_tools()
                print(f"🛠️ Available tools: {tools}")

                # Generate SQL using Gemini
                sql_prompt = f"""Tables: topics(id,name), problems(title,difficulty,topic_id,time_complexity)
                    The difficulty column contains text values: 'Easy', 'Medium', 'Hard'.
                    Question: "{question}"
                    Return ONLY valid Postgres SQL, no explanation, no markdown.
                    Use CASE WHEN to sort difficulty properly, e.g.:
                    ORDER BY CASE difficulty WHEN 'Hard' THEN 1 WHEN 'Medium' THEN 2 WHEN 'Easy' THEN 3 END ASC"""
                sql = generate_with_genai(sql_prompt, question, use_history=False).strip()
                # Strip markdown fences if Gemini wraps it
                sql = sql.replace("```sql", "").replace("```", "").strip()
                print(f"🔧 Generated SQL: {sql}")

                # Call the 'query' tool on the MCP server
                result = await session.call_tool("query", {"sql": sql})

                # Extract text from the result
                result_text = ""
                if hasattr(result, "content"):
                    for block in result.content:
                        if hasattr(block, "text"):
                            result_text += block.text
                else:
                    result_text = str(result)

        return f"✅ **SQL Executed:** `{sql}`\n📊 **Results:**\n{result_text[:1500]}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ MCP Error: {str(e)}"
    
def format_context_with_sources(matches: List) -> tuple[str, List[Dict]]:
    """
    Extract text from matches and track sources
    Returns: (combined_context, source_list)
    """
    pieces = []
    sources = []
    
    for idx, match in enumerate(matches, 1):
        # Extract metadata
        if isinstance(match, dict):
            meta = match.get("metadata") or {}
            score = match.get("score", 0)
            match_id = match.get("id", "unknown")
        else:
            meta = getattr(match, "metadata", {})
            score = getattr(match, "score", 0)
            match_id = getattr(match, "id", "unknown")
        
        # Extract text content
        if isinstance(meta, dict):
            text = meta.get("text") or meta.get("page_content") or meta.get("content")
            source_name = meta.get("source", "Unknown")
            page = meta.get("page", "N/A")
        else:
            text = getattr(meta, "text", None)
            source_name = getattr(meta, "source", "Unknown")
            page = getattr(meta, "page", "N/A")
        
        if text:
            pieces.append(f"[Source {idx}]\n{text}")
            sources.append({
                "id": idx,
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "score": float(score),
                "source": source_name,
                "page": page,
                "match_id": match_id
            })
        else:
            pieces.append(f"[Source {idx}]\n<no text available>")
            sources.append({
                "id": idx,
                "text_preview": "<no text>",
                "score": float(score),
                "source": "Unknown",
                "page": "N/A",
                "match_id": match_id
            })
    
    context = "\n\n---\n\n".join(pieces) if pieces else ""
    return context, sources

def route_question(question: str) -> str:
    """Smart keyword routing - FIXED"""
    question_lower = question.lower()
    
    # MCP: Data queries
    mcp_keywords = ['top', 'count', 'list', 'hardest', 'easiest', 'problems', 'stats', 'average', 'total', 'how many']
    if any(word in question_lower for word in mcp_keywords):
        return "MCP"
    
    # RAG: Theory questions  
    rag_keywords = ['explain', 'what is', 'how does', 'algorithm', 'complexity', 'works']
    if any(word in question_lower for word in rag_keywords):
        return "RAG"
    
    return "BOTH"


def generate_with_genai(
    context_text: str, 
    question: str, 
    use_history: bool = True
) -> str:
    """Generate answer using Google's Gemini API with conversation history"""
    if not client:
        return "❌ Error: GEMINI_API_KEY not configured in .env file"
    
    try:
        # Use the actual available models from the API
        model_names = [
            "models/gemini-flash-latest",      # Latest stable flash model
            "models/gemini-2.5-flash",         # Gemini 2.5 Flash
            "models/gemini-pro-latest",        # Latest stable pro model
            "models/gemini-2.5-pro",           # Gemini 2.5 Pro
            "models/gemini-2.0-flash",         # Gemini 2.0 Flash fallback
        ]
        
        # Build the prompt with history if enabled
        if use_history and conversation_history:
            history_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in conversation_history[-6:]  # Last 3 exchanges (6 messages)
            ])
            system_instruction = (
                "You are a Data Structure and Algorithm Expert.\n"
                "Answer based ONLY on the provided context. If the answer is not in the context, "
                "say 'I could not find the answer in the provided documents.'\n\n"
                f"Previous conversation:\n{history_text}\n\n"
                f"Context from documents:\n{context_text}\n\n"
                f"Current question: {question}\n\n"
                "Provide a clear, concise answer based on the context above."
            )
        else:
            system_instruction = (
                "You are a Data Structure and Algorithm Expert.\n"
                "Answer based ONLY on the provided context. If the answer is not in the context, "
                "say 'I could not find the answer in the provided documents.'\n\n"
                f"Context from documents:\n{context_text}\n\n"
                f"Question: {question}\n\n"
                "Provide a clear, concise answer based on the context above."
            )
        
        last_error = None
        for model_name in model_names:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=system_instruction,
                )
                
                # Extract text from response
                if response and hasattr(response, 'text') and response.text:
                    return response.text
                elif response and hasattr(response, 'candidates') and response.candidates:
                    if response.candidates[0].content.parts:
                        return response.candidates[0].content.parts[0].text
                else:
                    return "⚠️ No response generated from the model."
                    
            except Exception as model_error:
                error_str = str(model_error).lower()
                last_error = str(model_error)
                
                # Skip models that don't exist
                if "404" in error_str or "not found" in error_str:
                    continue
                
                # Skip rate-limited models and try next one
                if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    print(f"⚠️ {model_name} quota exceeded, trying next model...")
                    continue
                
                # For other errors, try next model
                continue
        
        # If all models failed
        if last_error and ("quota" in str(last_error).lower() or "429" in str(last_error)):
            return "❌ Rate limit exceeded on all models. Please wait a few minutes and try again."
        
        return f"❌ All models failed. Error: {last_error[:200]}"
        
    except Exception as e:
        return f"❌ Generation failed: {str(e)}"


async def search_pinecone(query_vector: List[float], top_k: int = 5) -> Optional[object]:
    """Search Pinecone for similar vectors"""
    if pinecone_index is None:
        print("⚠️ Pinecone not configured, skipping vector search")
        return None
    
    try:
        res = await asyncio.to_thread(
            pinecone_index.query,
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        return res
    except TypeError:
        # Fallback for different API versions
        try:
            res = await asyncio.to_thread(
                pinecone_index.query,
                query_vector,
                top_k,
                True
            )
            return res
        except Exception as e:
            print(f"❌ Pinecone search error: {e}")
            return None
    except Exception as e:
        print(f"❌ Pinecone search error: {e}")
        return None


def display_sources(sources: List[Dict]) -> None:
    """Display retrieved sources in a formatted way"""
    if not sources:
        return
    
    print("\n📚 Sources used:")
    print("=" * 70)
    for src in sources:
        print(f"  [{src['id']}] Score: {src['score']:.4f} | Source: {src['source']} | Page: {src['page']}")
        print(f"      Preview: {src['text_preview']}")
    print("=" * 70)


# ==================== Main Chat Function ====================

async def chatting(user_problem: str, show_sources: bool = True) -> None:
    """Main chatting function with improved features"""
    
    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": user_problem,
        "timestamp": datetime.now().isoformat()
    })
    print("\n🤔 Analyzing question type...")

    route = route_question(user_problem)
    print(f"📍 Routed to: {route}")

    answer_text = ""
    sources = []

    if route == "MCP":
        print("🔍 Querying DSA database via MCP...")
        answer_text = await mcp_query(user_problem)

    elif route == "RAG":
        print("📚 Searching DSA PDFs via Pinecone...")
        # Convert user query to embedding vector
        try:
            qvec = await embeddings.aembed_query(user_problem)
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            return

        # Normalize embedding shape
        if isinstance(qvec, list) and len(qvec) > 0 and isinstance(qvec[0], list):
            query_vector = qvec[0]
        else:
            query_vector = qvec

        # Search Pinecone for similar content
        res = await search_pinecone(query_vector, top_k=5)

        if res is None:
            print("⚠️ No search results available. Generating answer without context...")
            context = ""
            sources = []
        else:
            # Extract matches and create context
            matches = getattr(res, "matches", None) or (
                res.get("matches") if isinstance(res, dict) else None
            ) or []

            if not matches:
                print("⚠️ No relevant documents found in the database.")
                context = ""
                sources = []
            else:
                context, sources = format_context_with_sources(matches)
                answer_text = generate_with_genai(context, user_problem, use_history=True)
        
    else:  # BOTH - RAG + MCP
        print("🔄 Hybrid: PDFs + Database...")
        
        # RAG first
        try:
            qvec = await embeddings.aembed_query(user_problem)
            query_vector = qvec[0] if isinstance(qvec[0], list) else qvec
            rag_res = await search_pinecone(query_vector, top_k=3)
            
            if rag_res and rag_res.matches:
                rag_context, rag_sources = format_context_with_sources(rag_res.matches)
                sources = rag_sources
            else:
                rag_context = ""
        except:
            rag_context = ""
        
        # MCP second
        db_result = await mcp_query(user_problem)
        
        # Combine in Gemini
        combined_context = f"""
        RAG Context (theory): {rag_context}
        DB Results (data): {db_result}
                
        Answer comprehensively using both."""
        # Generate answer
        print("\n🤔 Thinking...")
        answer_text = generate_with_genai(combined_context, user_problem, use_history=True)
    
    # Add assistant response to history
    conversation_history.append({
        "role": "assistant",
        "content": answer_text,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display answer
    print("\n💡 Answer:")
    print("=" * 70)
    print(answer_text)
    print("=" * 70)
    
    # Display sources if requested
    if show_sources and sources:
        display_sources(sources)

    print(f"\n🔄 Route used: {route} | History length: {len(conversation_history)//2} exchanges")
# ==================== Main Loop ====================

async def main() -> None:
    """Main application loop"""
    print("\n" + "=" * 70)
    print("🤖 DSA Expert Chatbot - RAG System")
    print("=" * 70)
    print("\nCommands:")
    print("  • Type your question to get an answer")
    print("  • 'history' - View conversation history")
    print("  • 'clear' - Clear conversation history")
    print("  • 'exit/quit/q' - Exit the program")
    print("=" * 70 + "\n")
    
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\n📝 Ask me anything --> ")
            
            if not user_input.strip():
                continue
            
            user_input_lower = user_input.strip().lower()
            
            # Handle commands
            if user_input_lower in ("exit", "quit", "q"):
                print("\n👋 Goodbye! Have a great day!")
                break
            
            elif user_input_lower == "history":
                if not conversation_history:
                    print("\n📭 No conversation history yet.")
                else:
                    print("\n📜 Conversation History:")
                    print("=" * 70)
                    for msg in conversation_history:
                        role = "👤 You" if msg["role"] == "user" else "🤖 Assistant"
                        print(f"\n{role} ({msg['timestamp']}):")
                        print(msg["content"])
                    print("=" * 70)
                continue
            
            elif user_input_lower == "clear":
                conversation_history.clear()
                print("\n🧹 Conversation history cleared!")
                continue
            
            # Process the question
            await chatting(user_input, show_sources=True)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! (Interrupted by user)")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())