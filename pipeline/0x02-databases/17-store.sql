-- trigger that decreases the quantity of an item after adding a new order

CREATE TRIGGER update_items
  AFTER INSERT
  ON orders
  FOR EACH ROW
BEGIN
  UPDATE items SET quantity = quantity - new.number WHERE items.name=new.item_name;  
END