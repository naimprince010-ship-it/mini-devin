"""
Persistent Memory System for Plodder
Cross-session memory with vector storage and semantic search
"""

import os
import json
import pickle
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import logging

try:
    import sqlite3
    import aiosqlite
except ImportError:
    sqlite3 = None
    aiosqlite = None

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    np = None
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class PersistentMemory:
    """Persistent memory system with cross-session capabilities"""
    
    def __init__(self, storage_dir: str = "./memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Database paths
        self.db_path = self.storage_dir / "memory.db"
        self.vector_db_path = self.storage_dir / "vectors.pkl"
        self.skills_db_path = self.storage_dir / "skills.json"
        
        # Initialize components
        self.conn = None
        self.embeddings_model = None
        self.memory_vectors = {}
        self.skills_library = {}
        
        # Configuration
        self.max_memory_age_days = 30
        self.embedding_dim = 384  # For all-MiniLM-L6-v2 model
        
    async def initialize(self) -> bool:
        """Initialize the memory system"""
        try:
            # Initialize database
            await self._init_database()
            
            # Initialize embedding model
            await self._init_embedding_model()
            
            # Load existing data
            await self._load_memory_vectors()
            await self._load_skills_library()
            
            logger.info("Persistent memory system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            return False
    
    async def _init_database(self) -> None:
        """Initialize SQLite database for memory storage"""
        if not aiosqlite:
            logger.warning("aiosqlite not available, using file-based storage")
            return
            
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    importance_score REAL DEFAULT 0.0
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    tasks_completed TEXT,
                    context_summary TEXT
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    code TEXT,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    last_used TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.commit()
    
    async def _init_embedding_model(self) -> None:
        """Initialize sentence transformer model for embeddings"""
        if not SentenceTransformer:
            logger.warning("sentence-transformers not available, semantic search disabled")
            return
            
        try:
            # Use a lightweight model for embeddings
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
    
    async def _load_memory_vectors(self) -> None:
        """Load existing memory vectors from file"""
        if self.vector_db_path.exists():
            try:
                with open(self.vector_db_path, 'rb') as f:
                    self.memory_vectors = pickle.load(f)
                logger.info(f"Loaded {len(self.memory_vectors)} memory vectors")
            except Exception as e:
                logger.error(f"Failed to load memory vectors: {e}")
                self.memory_vectors = {}
    
    async def _load_skills_library(self) -> None:
        """Load existing skills library from file"""
        if self.skills_db_path.exists():
            try:
                with open(self.skills_db_path, 'r') as f:
                    self.skills_library = json.load(f)
                logger.info(f"Loaded {len(self.skills_library)} skills")
            except Exception as e:
                logger.error(f"Failed to load skills library: {e}")
                self.skills_library = {}
    
    async def add_memory(
        self, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        importance: float = 0.5
    ) -> str:
        """Add a new memory entry"""
        try:
            # Generate unique ID
            memory_id = hashlib.md5(f"{content}_{datetime.now().isoformat()}".encode()).hexdigest()
            
            # Create embedding if model is available
            embedding = None
            if self.embeddings_model:
                embedding = self.embeddings_model.encode([content])[0].tolist()
            
            # Store in database
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        INSERT INTO memories 
                        (id, content, metadata, tags, importance_score)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        memory_id,
                        content,
                        json.dumps(metadata or {}),
                        json.dumps(tags or []),
                        importance
                    ))
                    await db.commit()
            
            # Store in memory vectors
            if embedding is not None:
                self.memory_vectors[memory_id] = {
                    'content': content,
                    'embedding': embedding,
                    'metadata': metadata or {},
                    'tags': tags or [],
                    'created_at': datetime.now().isoformat(),
                    'importance': importance
                }
                await self._save_memory_vectors()
            
            logger.info(f"Added memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return ""
    
    async def search_memories(
        self, 
        query: str, 
        limit: int = 10,
        use_semantic_search: bool = True
    ) -> List[Dict[str, Any]]:
        """Search memories using text and semantic search"""
        try:
            results = []
            
            if use_semantic_search and self.embeddings_model and self.memory_vectors:
                # Semantic search using embeddings
                query_embedding = self.embeddings_model.encode([query])[0].tolist()
                
                # Calculate similarities
                similarities = []
                for memory_id, memory_data in self.memory_vectors.items():
                    memory_embedding = memory_data['embedding']
                    similarity = self._cosine_similarity(query_embedding, memory_embedding)
                    similarities.append((memory_id, similarity, memory_data))
                
                # Sort by similarity and get top results
                similarities.sort(key=lambda x: x[1], reverse=True)
                for memory_id, similarity, memory_data in similarities[:limit]:
                    results.append({
                        'id': memory_id,
                        'content': memory_data['content'],
                        'metadata': memory_data['metadata'],
                        'tags': memory_data['tags'],
                        'similarity': similarity,
                        'created_at': memory_data['created_at']
                    })
            
            # Also do text-based search as fallback
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute("""
                        SELECT id, content, metadata, tags, created_at, importance_score
                        FROM memories 
                        WHERE content LIKE ? OR tags LIKE ?
                        ORDER BY importance_score DESC, access_count DESC
                        LIMIT ?
                    """, (f"%{query}%", f"%{query}%", limit))
                    
                    text_results = await cursor.fetchall()
                    
                    for row in text_results:
                        memory_id, content, metadata, tags, created_at, importance = row
                        if not any(r['id'] == memory_id for r in results):
                            results.append({
                                'id': memory_id,
                                'content': content,
                                'metadata': json.loads(metadata),
                                'tags': json.loads(tags),
                                'similarity': 0.0,
                                'created_at': created_at,
                                'importance': importance
                            })
            
            # Update access statistics
            for result in results[:5]:  # Update top 5 results
                await self._update_memory_access(result['id'])
            
            logger.info(f"Found {len(results)} memories for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID"""
        try:
            if memory_id in self.memory_vectors:
                return self.memory_vectors[memory_id]
            
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute("""
                        SELECT id, content, metadata, tags, created_at, importance_score
                        FROM memories WHERE id = ?
                    """, (memory_id,))
                    
                    row = await cursor.fetchone()
                    if row:
                        memory_id, content, metadata, tags, created_at, importance = row
                        return {
                            'id': memory_id,
                            'content': content,
                            'metadata': json.loads(metadata),
                            'tags': json.loads(tags),
                            'created_at': created_at,
                            'importance': importance
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get memory {memory_id}: {e}")
            return None
    
    async def update_memory_importance(self, memory_id: str, importance: float) -> bool:
        """Update importance score of a memory"""
        try:
            if memory_id in self.memory_vectors:
                self.memory_vectors[memory_id]['importance'] = importance
                await self._save_memory_vectors()
            
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        UPDATE memories SET importance_score = ? WHERE id = ?
                    """, (importance, memory_id))
                    await db.commit()
            
            logger.info(f"Updated importance for memory {memory_id}: {importance}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory importance: {e}")
            return False
    
    async def add_skill(
        self, 
        name: str, 
        description: str, 
        code: str,
        category: str = "general"
    ) -> str:
        """Add a new skill to the skills library"""
        try:
            skill_id = hashlib.md5(f"{name}_{description}".encode()).hexdigest()
            
            skill = {
                'id': skill_id,
                'name': name,
                'description': description,
                'code': code,
                'category': category,
                'usage_count': 0,
                'success_rate': 0.0,
                'last_used': None,
                'created_at': datetime.now().isoformat()
            }
            
            self.skills_library[skill_id] = skill
            await self._save_skills_library()
            
            # Also store in database
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        INSERT OR REPLACE INTO skills 
                        (id, name, description, code, usage_count, success_rate, last_used, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        skill_id, name, description, code,
                        0, 0.0, None, skill['created_at']
                    ))
                    await db.commit()
            
            logger.info(f"Added skill: {name}")
            return skill_id
            
        except Exception as e:
            logger.error(f"Failed to add skill: {e}")
            return ""
    
    async def get_skill(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """Get a skill by name"""
        try:
            for skill_id, skill in self.skills_library.items():
                if skill['name'].lower() == skill_name.lower():
                    return skill
            
            # Search in database as fallback
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute("""
                        SELECT * FROM skills WHERE LOWER(name) = LOWER(?)
                    """, (skill_name,))
                    
                    row = await cursor.fetchone()
                    if row:
                        return {
                            'id': row[0],
                            'name': row[1],
                            'description': row[2],
                            'code': row[3],
                            'usage_count': row[4],
                            'success_rate': row[5],
                            'last_used': row[6],
                            'created_at': row[7]
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get skill {skill_name}: {e}")
            return None
    
    async def update_skill_usage(self, skill_id: str, success: bool = True) -> bool:
        """Update skill usage statistics"""
        try:
            if skill_id in self.skills_library:
                skill = self.skills_library[skill_id]
                skill['usage_count'] += 1
                
                # Update success rate
                if success:
                    current_success = skill['success_rate'] * (skill['usage_count'] - 1)
                    skill['success_rate'] = (current_success + 1) / skill['usage_count']
                
                skill['last_used'] = datetime.now().isoformat()
                await self._save_skills_library()
            
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        UPDATE skills 
                        SET usage_count = usage_count + 1,
                            success_rate = ?,
                            last_used = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (self.skills_library[skill_id]['success_rate'], skill_id))
                    await db.commit()
            
            logger.info(f"Updated usage for skill {skill_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update skill usage: {e}")
            return False
    
    async def start_session(self, session_id: str) -> bool:
        """Start a new session"""
        try:
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        INSERT OR REPLACE INTO sessions (id, start_time)
                        VALUES (?, CURRENT_TIMESTAMP)
                    """, (session_id,))
                    await db.commit()
            
            logger.info(f"Started session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            return False
    
    async def end_session(
        self, 
        session_id: str, 
        tasks_completed: List[str],
        context_summary: str
    ) -> bool:
        """End a session and save summary"""
        try:
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        UPDATE sessions 
                        SET end_time = CURRENT_TIMESTAMP,
                            tasks_completed = ?,
                            context_summary = ?
                        WHERE id = ?
                    """, (json.dumps(tasks_completed), context_summary, session_id))
                    await db.commit()
            
            logger.info(f"Ended session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
            return False
    
    async def cleanup_old_memories(self, days: int = None) -> int:
        """Clean up old memories to prevent storage bloat"""
        try:
            days = days or self.max_memory_age_days
            cutoff_date = datetime.now() - timedelta(days=days)
            
            deleted_count = 0
            
            # Clean up memory vectors
            old_memories = [
                memory_id for memory_id, memory_data in self.memory_vectors.items()
                if datetime.fromisoformat(memory_data['created_at']) < cutoff_date
            ]
            
            for memory_id in old_memories:
                del self.memory_vectors[memory_id]
                deleted_count += 1
            
            if deleted_count > 0:
                await self._save_memory_vectors()
            
            # Clean up database
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute("""
                        DELETE FROM memories WHERE created_at < ?
                    """, (cutoff_date.isoformat(),))
                    deleted_db = cursor.rowcount
                    await db.commit()
                    deleted_count += deleted_db
            
            logger.info(f"Cleaned up {deleted_count} old memories")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old memories: {e}")
            return 0
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not np:
            return 0.0
            
        try:
            v1, v2 = np.array(vec1), np.array(vec2)
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        except:
            return 0.0
    
    async def _update_memory_access(self, memory_id: str) -> None:
        """Update access statistics for a memory"""
        try:
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute("""
                        UPDATE memories 
                        SET accessed_at = CURRENT_TIMESTAMP,
                            access_count = access_count + 1
                        WHERE id = ?
                    """, (memory_id,))
                    await db.commit()
        except Exception as e:
            logger.error(f"Failed to update memory access: {e}")
    
    async def _save_memory_vectors(self) -> None:
        """Save memory vectors to file"""
        try:
            with open(self.vector_db_path, 'wb') as f:
                pickle.dump(self.memory_vectors, f)
        except Exception as e:
            logger.error(f"Failed to save memory vectors: {e}")
    
    async def _save_skills_library(self) -> None:
        """Save skills library to file"""
        try:
            with open(self.skills_db_path, 'w') as f:
                json.dump(self.skills_library, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save skills library: {e}")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        try:
            stats = {
                'total_memories': len(self.memory_vectors),
                'total_skills': len(self.skills_library),
                'storage_dir': str(self.storage_dir),
                'embedding_model_available': self.embeddings_model is not None,
                'database_available': aiosqlite is not None
            }
            
            if aiosqlite:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute("SELECT COUNT(*) FROM memories")
                    db_memories = (await cursor.fetchone())[0]
                    
                    cursor = await db.execute("SELECT COUNT(*) FROM sessions")
                    sessions = (await cursor.fetchone())[0]
                    
                    stats.update({
                        'database_memories': db_memories,
                        'total_sessions': sessions
                    })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}

# Example usage and helper functions
async def create_development_memory(
    memory_system: PersistentMemory,
    task_description: str,
    solution_code: str,
    lessons_learned: List[str]
) -> str:
    """Create a memory for a development task"""
    
    content = f"""
Task: {task_description}

Solution:
{solution_code}

Lessons Learned:
{chr(10).join(f"- {lesson}" for lesson in lessons_learned)}
"""
    
    metadata = {
        'task_type': 'development',
        'solution_code': solution_code,
        'lessons_learned': lessons_learned
    }
    
    tags = ['development', 'coding', 'solution'] + lessons_learned
    
    return await memory_system.add_memory(content, metadata, tags, importance=0.8)

async def create_reusable_skill(
    memory_system: PersistentMemory,
    skill_name: str,
    skill_description: str,
    implementation_code: str,
    category: str = "utility"
) -> str:
    """Create a reusable skill"""
    
    return await memory_system.add_skill(
        name=skill_name,
        description=skill_description,
        code=implementation_code,
        category=category
    )
