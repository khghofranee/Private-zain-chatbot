from sqlalchemy.orm import Session
from database import SessionLocal, get_password_hash, engine
from models import Admin
from models.Admin_model import Base

def init_admin():
    # Create tables first - this is the critical step
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Tables created successfully.")

    db = SessionLocal()
    try:
        # Check if admin already exists
        admin_exists = db.query(Admin).filter(Admin.username == "admin").first()
        if admin_exists:
            print("Admin user already exists")
            return

        # Create admin user
        print("Creating admin user...")
        admin = Admin(
            username="admin",
            email="admin@example.com",
            full_name="System Administrator",
            hashed_password=get_password_hash("adminpassword"),
            is_active=True
        )

        db.add(admin)
        db.commit()
        print("Admin user created successfully")

    except Exception as e:
        print(f"Error creating admin: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_admin()
