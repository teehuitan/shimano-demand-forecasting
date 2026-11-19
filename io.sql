SELECT
    MaterialCode,
    Product_Type_Code,
    Model_Code,
    Plan_Model_Code,
    Class_Text,
    Currency_Code,
    Customer_Country_Code,
    Unit_Price,
    Quantity,
    Ship_To_Code,
    Order_Status_Text,
    [Order_Entry_Date Date],
    [Customer_Request_Date Date],
    [Cancel_Date Date]
FROM Warehouse_AI.dbo.IncomingOrder_GoogleLooker
WHERE Plant = :plant