import streamlit as st
import sqlite3
from datetime import datetime

DATABASE_FILE = "todos.db"

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS todos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            priority TEXT DEFAULT 'medium',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed BOOLEAN DEFAULT FALSE
        )
    """)
    conn.commit()
    conn.close()

def add_todo(task, priority):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO todos (task, priority) VALUES (?, ?)",
        (task, priority)
    )
    conn.commit()
    conn.close()

def get_todos():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM todos ORDER BY completed ASC, created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_todo_status(todo_id, completed):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE todos SET completed = ? WHERE id = ?",
        (completed, todo_id)
    )
    conn.commit()
    conn.close()

def delete_todo(todo_id):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
    conn.commit()
    conn.close()

def main():
    st.set_page_config(page_title="Todo List", page_icon="✓")
    st.title("📝 Todo List Application")
    
    init_db()
    
    with st.form("todo_form", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            task_input = st.text_input("New Task", placeholder="What needs to be done?")
        with col2:
            priority = st.selectbox("Priority", ["low", "medium", "high"])
        submitted = st.form_submit_button("Add Task")
        
        if submitted and task_input:
            add_todo(task_input, priority)
            st.rerun()
    
    todos = get_todos()
    
    if not todos:
        st.info("No tasks yet. Add your first task above!")
    else:
        for todo in todos:
            todo_id, task, priority, created_at, completed = todo
            
            cols = st.columns([4, 1, 1, 1])
            
            with cols[0]:
                if completed:
                    st.success(f"✓ {task}")
                else:
                    st.warning(f"○ {task}")
            
            with cols[1]:
                priority_colors = {"low": "🟢", "medium": "🟡", "high": "🔴"}
                st.markdown(priority_colors.get(priority, "🟡"))
            
            with cols[2]:
                if st.button("✓" if not completed else "↩", key=f"toggle_{todo_id}"):
                    update_todo_status(todo_id, not completed)
                    st.rerun()
            
            with cols[3]:
                if st.button("🗑", key=f"delete_{todo_id}"):
                    delete_todo(todo_id)
                    st.rerun()
    
    st.divider()
    completed_count = sum(1 for t in todos if t[4])
    st.metric("Progress", f"{completed_count}/{len(todos)} tasks completed")

if __name__ == "__main__":
    main()
